"""Image attachment handling (facts for text-only servers, pixels for vision ones)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import base64
import binascii
import mimetypes
import re
import shutil
import struct
import textwrap

from ._optional import _PILImage, pytesseract
from .errors import BonsaiError

PNG_MAGIC = b'\x89PNG\r\n\x1a\n'

def sniff_format(data: bytes):
    """Identify a raster format from magic bytes (no third-party library needed)."""
    if data.startswith(PNG_MAGIC):
        return 'png'
    if data.startswith(b'\xff\xd8\xff'):
        return 'jpeg'
    if data[:6] in (b'GIF87a', b'GIF89a'):
        return 'gif'
    if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return 'webp'
    if data[:2] == b'BM':
        return 'bmp'
    return None

def png_dimensions(data):
    if len(data) < 26:
        return None
    w, h = struct.unpack('>II', data[16:24])
    depth, ctype = data[24], data[25]
    colortypes = {0: 'grayscale', 2: 'rgb', 3: 'indexed', 4: 'grayscale+alpha', 6: 'rgba'}
    return w, h, {'bit_depth': depth, 'color_type': colortypes.get(ctype, ctype)}

def jpeg_dimensions(data):
    i = 2
    n = len(data)
    while i + 9 < n:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        seg_len = struct.unpack('>H', data[i + 2:i + 4])[0]
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            h, w = struct.unpack('>HH', data[i + 5:i + 9])
            return w, h, {'components': data[i + 9] if i + 9 < n else None}
        i += 2 + seg_len
    return None

def gif_dimensions(data):
    if len(data) < 10:
        return None
    w, h = struct.unpack('<HH', data[6:10])
    return w, h, {}

def bmp_dimensions(data):
    if len(data) < 26:
        return None
    w, h = struct.unpack('<ii', data[18:26])
    return abs(w), abs(h), {}

def webp_dimensions(data):
    if len(data) < 30:
        return None
    fourcc = data[12:16]
    try:
        if fourcc == b'VP8X':
            w = int.from_bytes(data[24:27], 'little') + 1
            h = int.from_bytes(data[27:30], 'little') + 1
            return w, h, {'variant': 'extended'}
        if fourcc == b'VP8L':
            b0, b1, b2, b3 = data[21], data[22], data[23], data[24]
            w = ((b1 & 0x3F) << 8 | b0) + 1
            h = ((b3 & 0x0F) << 10 | b2 << 2 | (b1 & 0xC0) >> 6) + 1
            return w, h, {'variant': 'lossless'}
        if fourcc == b'VP8 ':
            w = struct.unpack('<H', data[26:28])[0] & 0x3FFF
            h = struct.unpack('<H', data[28:30])[0] & 0x3FFF
            return w, h, {'variant': 'lossy'}
    except (struct.error, IndexError):
        return None
    return None

def header_info(data: bytes):
    """Format + pixel dimensions straight from the file header, or (None, None, {})."""
    fmt = sniff_format(data)
    parsers = {'png': png_dimensions, 'jpeg': jpeg_dimensions, 'gif': gif_dimensions,
               'bmp': bmp_dimensions, 'webp': webp_dimensions}
    parser = parsers.get(fmt)
    if not parser:
        return None, None, {}
    res = parser(data)
    if not res:
        return fmt, None, {}
    w, h, extra = res
    return fmt, (w, h), extra

@dataclass
class ImageAttachment:
    path: Path
    fmt: str = ''
    mime: str = 'application/octet-stream'
    width: int = 0
    height: int = 0
    size_bytes: int = 0
    header_extra: dict = field(default_factory=dict)
    pillow: dict = field(default_factory=dict)
    ocr_text: str = ''
    data_url: str = ''
    payload_bytes: int = 0
    notes: list = field(default_factory=list)

    @property
    def display_name(self):
        return self.path.name

    def aspect(self):
        if not self.width or not self.height:
            return ''
        from math import gcd
        g = gcd(self.width, self.height) or 1
        w, h = self.width // g, self.height // g
        if max(w, h) > 40:
            return f'{self.width / self.height:.2f}:1'
        return f'{w}:{h}'

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path, max_side=1024, use_ocr=False, encode=True):
        p = Path(path).expanduser()
        if not p.is_file():
            raise BonsaiError(f'image not found: {p}')
        data = p.read_bytes()
        fmt, dims, extra = header_info(data)
        if fmt is None:
            mime = mimetypes.guess_type(p.name)[0] or ''
            if not mime.startswith('image/'):
                raise BonsaiError(
                    f'{p.name}: not a recognised image (no PNG/JPEG/GIF/WebP/BMP magic bytes, '
                    f'mime={mime or "unknown"}).')
            fmt = mime.split('/', 1)[1]
        mime = {'jpg': 'image/jpeg'}.get(fmt, 'image/' + fmt)
        att = cls(path=p, fmt=fmt, mime=mime, size_bytes=len(data), header_extra=extra)
        if dims:
            att.width, att.height = dims
        att.pillow = cls._pillow_stats(p)
        if att.pillow.get('size') and not att.width:
            att.width, att.height = att.pillow['size']
        if use_ocr:
            att.ocr_text = cls._ocr(p)
        if encode:
            att.data_url, att.payload_bytes = cls._encode(p, data, att.width, att.height, max_side)
        return att

    @staticmethod
    def _pillow_stats(path):
        if _PILImage is None:
            return {}
        try:
            with _PILImage.open(path) as im:
                info = {'size': im.size, 'mode': im.mode, 'format': im.format}
                try:
                    orient = im.getexif().get(274)
                    if orient:
                        info['exif_orientation'] = orient
                except Exception:
                    pass
                try:
                    small = im.convert('RGB').resize((16, 16))
                    px = list(small.getdata())
                    n = len(px) or 1
                    mean = tuple(sum(c[i] for c in px) // n for i in range(3))
                    info['mean_rgb'] = '#%02x%02x%02x' % mean
                    quant = im.convert('RGB').quantize(colors=4, method=_PILImage.Quantize.MEDIANCUT)
                    palette = [c[:3] for c in quant.getpalette()[:12]] if quant.getpalette() else []
                    info['dominant_rgb'] = ['#%02x%02x%02x' % tuple(c) for c in palette[:4] if c]
                except Exception:
                    pass
                return info
        except Exception as e:
            return {'error': str(e)}

    @staticmethod
    def _ocr(path):
        if pytesseract is None or not shutil.which('tesseract'):
            return ''
        if _PILImage is None:
            return ''
        try:
            with _PILImage.open(path) as im:
                text = pytesseract.image_to_string(im.convert('RGB'))
            return re.sub(r'\n{3,}', '\n\n', text).strip()[:2000]
        except Exception as e:
            return f'[OCR failed: {e}]'

    @staticmethod
    def _encode(path, data, width, height, max_side):
        """Base64 data URL, downscaled when Pillow is available and the image is big."""
        if _PILImage is not None and max_side and (max(width, height) > max_side or len(data) > 1_500_000):
            try:
                import io
                with _PILImage.open(path) as im:
                    im = im.convert('RGB') if im.mode in ('P', 'LA', 'RGBA', 'CMYK') else im
                    im.thumbnail((max_side, max_side))
                    buf = io.BytesIO()
                    im.save(buf, format='JPEG', quality=85, optimize=True)
                    payload = buf.getvalue()
                if len(payload) < len(data):
                    b64 = base64.b64encode(payload).decode()
                    return 'data:image/jpeg;base64,' + b64, len(payload)
            except Exception:
                pass
        try:
            b64 = base64.b64encode(data).decode()
        except (binascii.Error, ValueError) as e:
            raise BonsaiError(f'could not encode {path}: {e}') from None
        fmt = sniff_format(data) or 'jpeg'
        mime = 'image/jpeg' if fmt == 'jpg' else 'image/' + fmt
        return f'data:{mime};base64,' + b64, len(data)

    # ------------------------------------------------------------------
    def message_part(self):
        return {'type': 'image_url', 'image_url': {'url': self.data_url}}

    def text_card(self, vision):
        """What the model is actually told about the file.

        With a vision projector the pixels go through too, so this is a short caption.
        Without one (the default text-only deployment) this card is the *only* thing the
        model receives, so it carries every fact the client could measure.
        """
        lines = [f'[Image attached: {self.display_name}]']
        dims = f'{self.width}x{self.height} px' if self.width else 'dimensions unknown'
        extra = ''
        if self.header_extra:
            bits = self.header_extra.get('bit_depth')
            ct = self.header_extra.get('color_type')
            variant = self.header_extra.get('variant')
            extra = ', '.join(x for x in (
                f'{bits}-bit/channel' if bits else '',
                str(ct) if ct else '',
                f'{variant} variant' if variant else '') if x)
        lines.append(f'- file: {self.fmt.upper()}, {dims}'
                     + (f', aspect {self.aspect()}' if self.aspect() else '')
                     + f', {self.size_bytes / 1024:.1f} KiB on disk'
                     + (f' ({extra})' if extra else ''))
        if self.pillow:
            if 'error' in self.pillow:
                lines.append(f'- Pillow could not decode it: {self.pillow["error"]}')
            else:
                bits = []
                if self.pillow.get('mode'):
                    bits.append('mode ' + str(self.pillow['mode']))
                if self.pillow.get('mean_rgb'):
                    bits.append('mean colour ' + str(self.pillow['mean_rgb']))
                if self.pillow.get('dominant_rgb'):
                    bits.append('dominant ' + ', '.join(self.pillow['dominant_rgb']))
                if self.pillow.get('exif_orientation'):
                    bits.append('EXIF orientation ' + str(self.pillow['exif_orientation']))
                if bits:
                    lines.append('- decoded (Pillow): ' + '; '.join(bits))
        else:
            lines.append('- Pillow is not installed here, so no pixel statistics were extracted '
                         '(pip install pillow for mean/dominant colours).')
        if self.ocr_text:
            lines.append('- OCR text (tesseract):\n' + textwrap.indent(self.ocr_text, '    '))
        elif pytesseract is None or not shutil.which('tesseract'):
            lines.append('- OCR: unavailable (needs pytesseract + the tesseract binary; rerun with --ocr).')
        if vision:
            lines.append('- the pixels are attached to this message; the server has a vision projector.')
        else:
            lines.append('- NOTE: this server is TEXT-ONLY (no vision projector), so the model cannot '
                         'see the image. The measurements above are everything it has.')
        return '\n'.join(lines)

def load_attachments(paths, max_side=1024, use_ocr=False):
    return [ImageAttachment.load(p, max_side=max_side, use_ocr=use_ocr) for p in paths]

def build_user_message(text, attachments, vision):
    """Compose the user message: multimodal parts when the server can see, text card otherwise."""
    if not attachments:
        return {'role': 'user', 'content': text or ''}
    if vision:
        parts = [{'type': 'text', 'text': text or 'Describe this image.'}]
        parts += [a.message_part() for a in attachments]
        return {'role': 'user', 'content': parts}
    cards = '\n\n'.join(a.text_card(vision=False) for a in attachments)
    body = (text or 'What can you tell me about this file?')
    return {'role': 'user', 'content': body + '\n\n' + cards}
