"""
Bot de Pré-Atendimento — Suporte Loving
========================================

Bot do Telegram com IA conversacional alimentada pela API oficial da Anthropic.

COMO FUNCIONA
-------------
1. O bot carrega a base de conhecimento de `knowledge_base.md` (políticas, FAQ, etc.)
2. Quando o cliente envia mensagem, o Claude responde com base APENAS nessa base
3. Se a dúvida for resolvida pela base → o bot responde e encerra cordialmente
4. Se NÃO for resolvida (cancelamento, reembolso, bug, conta etc.) → o Claude
   coleta nome, e-mail e detalhes do problema durante a conversa, e usa a
   ferramenta `escalate_to_human`. Nesse momento o bot mostra UM ÚNICO BOTÃO
   "Abrir e-mail" — todo o resumo já vai dentro do mailto, sem aparecer no chat.

VARIÁVEIS DE AMBIENTE
---------------------
TELEGRAM_BOT_TOKEN  (obrigatório) Token do @BotFather
ANTHROPIC_API_KEY   (obrigatório) Chave do console.anthropic.com
ANTHROPIC_MODEL     (opcional)    Padrão: claude-haiku-4-5-20251001
RENDER_EXTERNAL_URL (auto)        Definida pelo Render. Se presente, ativa modo webhook.
PORT                (auto)        Definida pelo Render. Porta onde o webhook escuta.
WEBHOOK_SECRET      (opcional)    Token secreto para validar requisições do Telegram.
"""

import logging
import os
from urllib.parse import quote

from anthropic import Anthropic
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

SUPPORT_EMAIL = "lovingsuporte@gmail.com"
BOT_NAME = "Suporte Loving"
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
MAX_HISTORY_TURNS = 20  # quantas mensagens manter por usuário (evita estourar contexto)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base de conhecimento
# ---------------------------------------------------------------------------

def load_knowledge_base() -> str:
    """Lê a base de conhecimento do arquivo knowledge_base.md ao lado do bot."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "knowledge_base.md")
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        logger.info("Base de conhecimento carregada (%d caracteres)", len(content))
        return content
    except FileNotFoundError:
        logger.warning("knowledge_base.md não encontrado — bot funcionará sem políticas")
        return "(Base de conhecimento ainda não fornecida pelo administrador.)"


KNOWLEDGE_BASE = load_knowledge_base()

SYSTEM_PROMPT = f"""Você é o assistente virtual do **{BOT_NAME}**, fazendo pré-atendimento humanizado em português brasileiro.

# SEU PAPEL

1. Cumprimentar o cliente com empatia, em frases curtas e tom acolhedor.
2. Entender a dúvida ou problema dele.
3. Tentar RESOLVER usando APENAS as informações da Base de Conhecimento abaixo.
4. Se a base de conhecimento responde a dúvida → responda diretamente, com clareza, e encerre cordialmente.
5. Se a base NÃO resolve, OU se o caso exige ação humana (cancelamento, reembolso,
   bug técnico, problema de conta/pagamento, dúvida fora do escopo da base, ou cliente pedindo atendente humano) → você precisa:
     a) Conversar com o cliente para entender o problema em detalhes
     b) Coletar **nome completo** e **e-mail**
     c) Quando tiver tudo, chamar a ferramenta `escalate_to_human` com os dados estruturados

# REGRAS IMPORTANTES

- Tom humano, gentil, sem formalidade exagerada. Frases curtas. No máximo 1 emoji por mensagem.
- NUNCA invente informações que não estão na base de conhecimento.
- NÃO peça nome e e-mail logo de cara — só quando ficar claro que o caso precisa ser escalado.
- Se o cliente disser que é urgente ou crítico, escale imediatamente (urgency = Alta).
- Em respostas resolvidas pela base, ofereça no fim "se precisar de mais alguma coisa, é só me chamar".
- Se a pergunta for totalmente fora do escopo (não é sobre a Loving), explique gentilmente que você só pode ajudar com assuntos da Loving.

# BASE DE CONHECIMENTO

{KNOWLEDGE_BASE}

# FIM DA BASE DE CONHECIMENTO
"""

ESCALATE_TOOL = {
    "name": "escalate_to_human",
    "description": (
        "Encaminha o cliente para a equipe humana de suporte por e-mail. "
        "Use SOMENTE quando: a base de conhecimento não resolveu a questão, OU "
        "o cliente pediu atendente humano, OU o caso exige ação manual "
        "(cancelamento, reembolso, bug, problema de conta/pagamento). "
        "Antes de chamar, você precisa ter conversado o suficiente para ter "
        "nome, e-mail e descrição clara do problema."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Nome completo do cliente",
            },
            "email": {
                "type": "string",
                "description": "E-mail do cliente para retorno",
            },
            "title": {
                "type": "string",
                "description": "Título curto do problema (até 60 caracteres)",
            },
            "summary": {
                "type": "string",
                "description": "Resumo objetivo do problema em 1 a 2 frases",
            },
            "full_description": {
                "type": "string",
                "description": "Descrição completa nas palavras do cliente, juntando o que ele disse durante a conversa",
            },
            "category": {
                "type": "string",
                "enum": [
                    "Acesso/Login",
                    "Pagamento/Cobrança",
                    "Bug ou Erro Técnico",
                    "Dúvida sobre Uso",
                    "Cancelamento/Reembolso",
                    "Sugestão",
                    "Outro",
                ],
            },
            "urgency": {
                "type": "string",
                "enum": ["Baixa", "Média", "Alta"],
            },
        },
        "required": [
            "name",
            "email",
            "title",
            "summary",
            "full_description",
            "category",
            "urgency",
        ],
    },
}


# ---------------------------------------------------------------------------
# Cliente Anthropic e estado
# ---------------------------------------------------------------------------

anthropic_client = Anthropic()

# histórico por chat_id (em memória — se reiniciar, zera)
conversations: dict[int, list] = {}


def trim_history(history: list) -> list:
    """Mantém só as últimas N mensagens para não estourar contexto/custo."""
    if len(history) > MAX_HISTORY_TURNS * 2:
        return history[-MAX_HISTORY_TURNS * 2:]
    return history


def build_email_body(data: dict) -> str:
    return "\n".join([
        "=== PRÉ-ATENDIMENTO SUPORTE LOVING ===",
        "",
        f"Nome: {data['name']}",
        f"E-mail do cliente: {data['email']}",
        f"Motivo: {data['title']}",
        f"Categoria: {data['category']}",
        f"Urgência: {data['urgency']}",
        "",
        "----- Descrição completa -----",
        data["full_description"],
        "",
        "----- Resumo automático -----",
        data["summary"],
        "",
        "----- Origem -----",
        "Atendimento iniciado via bot do Telegram + IA Anthropic.",
    ])


# ---------------------------------------------------------------------------
# Handlers do Telegram
# ---------------------------------------------------------------------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    conversations[chat_id] = []
    await update.message.reply_text(
        f"Oi! 👋 Aqui é o assistente do {BOT_NAME}. Em que posso te ajudar hoje?"
    )


async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    conversations.pop(chat_id, None)
    await update.message.reply_text(
        "Atendimento encerrado. Quando precisar, é só digitar /start. 👋"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    user_text = (update.message.text or "").strip()
    if not user_text:
        return

    history = conversations.setdefault(chat_id, [])
    history.append({"role": "user", "content": user_text})
    history = trim_history(history)
    conversations[chat_id] = history

    # mostra "digitando..." enquanto o Claude pensa
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    try:
        response = anthropic_client.messages.create(
            model=MODEL,
            max_tokens=1024,
            # cache_control no system → barato e rápido em chamadas seguintes
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=[ESCALATE_TOOL],
            messages=history,
        )
    except Exception:
        logger.exception("Erro ao chamar a API da Anthropic")
        history.pop()  # remove a mensagem do usuário pra ele poder reenviar
        await update.message.reply_text(
            "Tive um probleminha técnico aqui 😅. Pode tentar de novo em instantes?"
        )
        return

    # Separa texto e tool_use da resposta
    text_parts: list[str] = []
    tool_use_block = None
    for block in response.content:
        if block.type == "text" and block.text.strip():
            text_parts.append(block.text.strip())
        elif block.type == "tool_use" and block.name == "escalate_to_human":
            tool_use_block = block

    reply_text = "\n\n".join(text_parts).strip()

    # Salva o turno do assistente no histórico (formato cru aceito pela API)
    history.append(
        {
            "role": "assistant",
            "content": [block.model_dump() for block in response.content],
        }
    )

    if tool_use_block is not None:
        # Caso de escalada → mostra SOMENTE o botão de e-mail
        data = tool_use_block.input
        body = build_email_body(data)
        subject = f"[Suporte] {data.get('title', 'Solicitação')}"
        mailto = (
            f"mailto:{SUPPORT_EMAIL}"
            f"?subject={quote(subject)}"
            f"&body={quote(body)}"
        )
        keyboard = InlineKeyboardMarkup(
            [[InlineKeyboardButton("📧 Abrir e-mail e enviar", url=mailto)]]
        )

        intro = reply_text or (
            "Tudo certo! Já organizei seu atendimento. "
            "Toque no botão abaixo — ele abre seu app de e-mail com tudo pronto, "
            "é só apertar enviar. Nossa equipe te responde em breve. 💙"
        )
        await update.message.reply_text(intro, reply_markup=keyboard)

        # Devolve o resultado da ferramenta pro Claude (mantém o histórico válido)
        history.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_block.id,
                        "content": (
                            "Ticket apresentado ao cliente. "
                            "Botão de e-mail exibido. Encerrar a conversa cordialmente "
                            "se ele falar de novo, a menos que abra um caso novo."
                        ),
                    }
                ],
            }
        )
    else:
        # Resposta normal de texto
        if reply_text:
            await update.message.reply_text(reply_text)
        else:
            await update.message.reply_text(
                "Pode me contar um pouquinho mais? Quero entender direito pra te ajudar. 🙂"
            )


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("Defina TELEGRAM_BOT_TOKEN com o token do BotFather.")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Defina ANTHROPIC_API_KEY com a chave do console.anthropic.com.")

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("cancel", cmd_cancel))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    public_url = os.environ.get("RENDER_EXTERNAL_URL")

    if public_url:
        # ----- Produção (Render) — modo WEBHOOK -----
        port = int(os.environ.get("PORT", "10000"))
        url_path = "telegram-webhook"
        webhook_url = f"{public_url.rstrip('/')}/{url_path}"
        # Usa parte do token como secret se não houver outro definido
        secret_token = os.environ.get("WEBHOOK_SECRET", token.split(":")[1][:32])

        logger.info(
            "%s online em modo WEBHOOK | url=%s | modelo=%s",
            BOT_NAME, webhook_url, MODEL,
        )
        app.run_webhook(
            listen="0.0.0.0",
            port=port,
            url_path=url_path,
            webhook_url=webhook_url,
            secret_token=secret_token,
            allowed_updates=Update.ALL_TYPES,
        )
    else:
        # ----- Local / desenvolvimento — modo POLLING -----
        logger.info(
            "%s online em modo POLLING (local) | modelo=%s",
            BOT_NAME, MODEL,
        )
        app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
