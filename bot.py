"""
Bot de Pré-Atendimento — Suporte Loving
========================================

Bot do Telegram com IA conversacional alimentada pela API oficial da Anthropic.

COMO FUNCIONA
-------------
1. O bot carrega a base de conhecimento de `knowledge_base.md` (políticas, FAQ, etc.)
2. Quando o cliente envia mensagem, o Claude responde com base APENAS nessa base
3. Se a dúvida for resolvida pela base → o bot responde e encerra cordialmente
4. Se NÃO for resolvida (ou se o cliente pedir humano) → o bot informa o e-mail
   de suporte e o prazo de retorno, sem coletar dados nem mostrar botão.

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

from anthropic import Anthropic
from telegram import Update
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
RESPONSE_SLA = "1 a 2 dias úteis"
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

# FLUXO PADRÃO

1. **Cumprimentar** o cliente com empatia, em frases curtas e tom acolhedor.
2. **Entender** a dúvida ou problema do cliente — pergunte com calma o que ele precisa.
3. **Tentar RESOLVER** usando APENAS as informações da Base de Conhecimento abaixo.
4. Se a base de conhecimento responde → responda diretamente, com clareza, e encerre cordialmente oferecendo "se precisar de mais alguma coisa, é só me chamar 🙂".

# QUANDO PRECISAR ESCALAR PARA HUMANO

Você precisa escalar para humano nestes casos:
- A base de conhecimento não cobre o assunto (ex: saque, reembolso, cobrança específica, bug técnico)
- O caso exige ação manual da equipe (cancelar conta, alterar dados sensíveis, reembolsar)
- O cliente pediu explicitamente para falar com atendente humano

**O fluxo de escalada tem DUAS ETAPAS — você deve seguir as duas, nunca pular nenhuma:**

## ETAPA 1 — CONFIRMAÇÃO

Antes de passar o e-mail, você precisa CONFIRMAR com o cliente sobre qual assunto ele quer atendimento humano. O objetivo é deixar claro pra ele que você entendeu o caso. Faça assim:

> "Perfeito[, NOME se ele tiver dito]! Só deixa eu confirmar pra te encaminhar direitinho: você gostaria de falar com um atendente humano sobre **[assunto resumido em 1-3 palavras]**, certo? É isso mesmo?"

Use o primeiro nome do cliente APENAS se ele tiver mencionado naturalmente durante a conversa. NUNCA peça o nome dele — se ele não disse, é só não usar.

NÃO peça e-mail. NÃO peça telefone. NÃO peça nenhum dado.

Aguarde a resposta do cliente.

## ETAPA 2 — RESPOSTA FINAL (depois que ele confirmar com sim/correto/isso/etc.)

Quando o cliente confirmar, dê a resposta padrão abaixo, adaptando o tom mas mantendo o conteúdo:

> "Então[, NOME], olha só: o nosso atendimento humanizado é feito pelo e-mail **{SUPPORT_EMAIL}**, e damos de {RESPONSE_SLA} para retorno. É só mandar um e-mail por lá explicando o seu caso de [assunto] que a equipe vai te ajudar com calma. Tudo bem? 💙"

Depois disso, encerre cordialmente. Se ele perguntar mais coisas sobre o mesmo assunto, reforce gentilmente que o caminho é pelo e-mail.

# REGRAS GERAIS

- Tom humano, gentil, próximo (nunca "prezado(a)"). Frases curtas. No máximo 1 emoji por mensagem.
- NUNCA invente informações que não estão na base de conhecimento.
- NUNCA peça dados pessoais do cliente (nome, e-mail, telefone, CPF, etc.). Se ele dizer espontaneamente, ok usar o primeiro nome em respostas, mas nunca solicitar.
- Se a pergunta for fora do escopo (não é sobre a Loving), explique gentilmente e siga o mesmo fluxo de escalada (confirmar → e-mail).
- O retorno SEMPRE é pelo e-mail — nunca prometa retorno pelo Telegram.

# BASE DE CONHECIMENTO

{KNOWLEDGE_BASE}

# FIM DA BASE DE CONHECIMENTO
"""


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
            messages=history,
        )
    except Exception:
        logger.exception("Erro ao chamar a API da Anthropic")
        history.pop()  # remove a mensagem do usuário pra ele poder reenviar
        await update.message.reply_text(
            "Tive um probleminha técnico aqui 😅. Pode tentar de novo em instantes?"
        )
        return

    # Extrai o texto de resposta
    text_parts: list[str] = []
    for block in response.content:
        if block.type == "text" and block.text.strip():
            text_parts.append(block.text.strip())

    reply_text = "\n\n".join(text_parts).strip()

    # Salva o turno do assistente no histórico
    history.append(
        {
            "role": "assistant",
            "content": [block.model_dump() for block in response.content],
        }
    )

    if reply_text:
        await update.message.reply_text(reply_text, parse_mode="Markdown")
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
