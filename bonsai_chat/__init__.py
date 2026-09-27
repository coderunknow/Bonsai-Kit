"""Bonsai-Kit chat client — a standard-library client for a Ternary Bonsai 2 27B
OpenAI-compatible endpoint.

The package is the implementation; ``bonsai_chat.py`` at the repository root is a thin
shim so the documented ``python3 bonsai_chat.py ...`` invocation keeps working from a
bare checkout with no install step.

v0.6.0 split the single 4,754-line v0.5.0 client into these modules so each layer can
be tested and reused in isolation. Every name the v0.5.0 module exposed at the top
level is still reachable as ``bonsai_chat.<name>``.

v0.6.0 also added, on top of that split: ``Config`` (layered configuration with
inspectable precedence), ``BranchTree`` (conversation branching), ``EndpointRegistry``
(multi-endpoint with a per-endpoint capability map), ``Budget`` (token/turn ceilings),
``register_mcp`` (MCP tools through the existing registry), ``export`` (JSON/HTML) and
``bonsai_chat.api.Bonsai`` (the documented importable surface).

Stable programmatic surface (see ``bonsai_chat.api`` for the documented wrapper):

    BonsaiClient(base_url, api_key, model=DEFAULT_MODEL, ...)
        .chat(messages, **params) -> dict
        .stream_chat(messages, **params) -> ChatStream
        .models() / .props() / .health() / .tokenize(text)
    ChatStream    — context-managed; iterate ``.events()`` or call ``.text()``
    CapabilityMap / ReasoningConfig / Settings / Conversation / Agent / ToolRegistry
    run_doctor(client, ...) / run_benchmark(client, ...) / run_selftest(...)
    main(argv)    — the CLI entry point
"""

from ._meta import DEFAULT_MODEL, VERSION
from ._optional import _PILImage, optional_features, pygments, pytesseract

from .errors import BonsaiAPIError, BonsaiError, CancelledByUser, CapabilityError, StreamInterrupted, TransportError
from .style import ANSI_RE, Style, strip_ansi, visible_len, wrap_ansi
from .sse import SSEDecoder, accumulate_tool_calls, arguments_complete, iter_sse_events, normalize_base_url
from .transport import HttpTransport
from .reasoning import EFFORT_ALIASES, EFFORT_UNRELIABLE, EFFORT_VALUES, REASONING_DISPLAYS, ReasoningConfig, THINK_ALIASES, THINK_BUDGETS, THINK_ORDER, parse_effort, parse_think_level
from .capabilities import CapabilityMap
from .client import BonsaiClient
from .streaming import ChatStream
from .tokens import TokenCounter, estimate_tokens
from .markdown import HEADING_RE, HR_RE, INLINE_RE, LIST_RE, MarkdownWriter, QUOTE_RE, TABLE_ROW_RE, TABLE_SEP_RE, highlight_code, html_unescape, render_markdown
from .images import ImageAttachment, PNG_MAGIC, bmp_dimensions, build_user_message, gif_dimensions, header_info, jpeg_dimensions, load_attachments, png_dimensions, sniff_format, webp_dimensions
from .tools import Tool, ToolContext, ToolError, ToolRegistry, ToolTimeout
from .conversation import Conversation, DEFAULT_SYSTEM, SESSION_SCHEMA, Settings, TurnResult, TurnStats, WIRE_ROLES, mask_key
from .agent import Agent
from .render import LiveRenderer, format_stats, format_stats_block
from .benchmark import BENCH_PROMPT_BLOCKS, render_benchmark, run_benchmark, sig, summarize
from .doctor import DOCTOR_STATES, run_doctor
from .app import ChatApp, HELP_TEXT
from .selftest import make_png, run_selftest
from .cli import build_parser, main, make_app
from .config import Config, deep_merge, find_config, load_config, save_config, write_default_config
from .budget import Budget
from .branches import Branch, BranchTree, MAIN as BRANCH_MAIN
from .endpoints import Endpoint, EndpointRegistry
from .export import export, export_html, export_json
from .mcp import McpError, McpServer, close_all as mcp_close_all, register_mcp
from .api import Bonsai, Result, read_batch, run_batch, write_batch

__all__ = [
    'VERSION', 'DEFAULT_MODEL', '_PILImage', 'optional_features', 'pygments',
    'pytesseract', 'BonsaiAPIError', 'BonsaiError', 'CancelledByUser',
    'CapabilityError', 'StreamInterrupted', 'TransportError', 'ANSI_RE', 'Style',
    'strip_ansi', 'visible_len', 'wrap_ansi', 'SSEDecoder', 'accumulate_tool_calls',
    'arguments_complete', 'iter_sse_events', 'normalize_base_url', 'HttpTransport',
    'EFFORT_ALIASES', 'EFFORT_UNRELIABLE', 'EFFORT_VALUES', 'REASONING_DISPLAYS',
    'ReasoningConfig', 'THINK_ALIASES', 'THINK_BUDGETS', 'THINK_ORDER', 'parse_effort',
    'parse_think_level', 'CapabilityMap', 'BonsaiClient', 'ChatStream', 'TokenCounter',
    'estimate_tokens', 'HEADING_RE', 'HR_RE', 'INLINE_RE', 'LIST_RE', 'MarkdownWriter',
    'QUOTE_RE', 'TABLE_ROW_RE', 'TABLE_SEP_RE', 'highlight_code', 'html_unescape',
    'render_markdown', 'Config', 'deep_merge', 'find_config', 'load_config',
    'save_config', 'write_default_config', 'Budget', 'Branch', 'BranchTree',
    'BRANCH_MAIN', 'Endpoint', 'EndpointRegistry', 'export', 'export_html',
    'export_json', 'McpError', 'McpServer', 'mcp_close_all', 'register_mcp', 'Bonsai',
    'Result', 'run_batch', 'read_batch', 'write_batch',
    'ImageAttachment', 'PNG_MAGIC', 'bmp_dimensions',
    'build_user_message', 'gif_dimensions', 'header_info', 'jpeg_dimensions',
    'load_attachments', 'png_dimensions', 'sniff_format', 'webp_dimensions', 'Tool',
    'ToolContext', 'ToolError', 'ToolRegistry', 'ToolTimeout', 'Conversation',
    'DEFAULT_SYSTEM', 'SESSION_SCHEMA', 'Settings', 'TurnResult', 'TurnStats',
    'WIRE_ROLES', 'mask_key', 'Agent', 'LiveRenderer', 'format_stats',
    'format_stats_block', 'BENCH_PROMPT_BLOCKS', 'render_benchmark', 'run_benchmark',
    'sig', 'summarize', 'DOCTOR_STATES', 'run_doctor', 'ChatApp', 'HELP_TEXT',
    'make_png', 'run_selftest', 'build_parser', 'main', 'make_app'
]
