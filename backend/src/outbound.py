import asyncio
import logging
import os
import uuid

import aiohttp
from dotenv import load_dotenv
from livekit import api, rtc
from livekit.agents import (
    Agent,
    AgentSession,
    room_io,
    tokenize,
)
from livekit.plugins import deepgram, murf, noise_cancellation, openai, silero

logger = logging.getLogger("outbound")

load_dotenv(".env.local")

# DAY 6 OUTBOUND SYSTEM PROMPT
OUTBOUND_SYSTEM_PROMPT = """
IDENTITY: You are Anisha, a voice assistant for 'Bol-Khata' (Local Commerce track).
OBJECTIVES: Act as an outbound stock reminder assistant.

STRICT OPENING SCRIPT (MANDATORY):
In an outbound call, you speak first. You MUST immediately say the following exact things in your first sentence as soon as the call connects:
- "Hello, I am Anisha from Bol-Khata."
- "I am calling to remind you that your potato stock might be running low based on your past orders."
- "If you do not want to receive these reminder calls, just tell me to stop."

STRICT LANGUAGE & SCRIPT RULES:
1. STRICT 1-to-1 language matching:
   - If the user speaks English, reply ONLY in pure English. No Hindi or Hinglish.
   - If the user speaks Hindi, reply ONLY in pure Hindi using Devanagari script (e.g., नमस्ते). No English, Hinglish, or romanized Hindi.
   - Do NOT mix languages in a single response. Do NOT translate your response or add translations.
2. Keep responses short and conversational.
"""


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=OUTBOUND_SYSTEM_PROMPT)


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    url = os.environ.get("LIVEKIT_URL")
    api_key = os.environ.get("LIVEKIT_API_KEY")
    api_secret = os.environ.get("LIVEKIT_API_SECRET")

    sip_trunk_id = os.environ.get("SIP_TRUNK_ID")
    my_sip_uri = os.environ.get("MY_SIP_URI")

    if not all([url, api_key, api_secret]):
        logger.error("Missing required LiveKit environment variables.")
        return

    if not sip_trunk_id or not my_sip_uri:
        logger.error("Missing SIP_TRUNK_ID or MY_SIP_URI.")
        return

    # Generate a unique room name for this outbound call
    room_name = f"outbound-call-{uuid.uuid4().hex[:8]}"
    room = rtc.Room()

    # Create a token for the agent to join the room
    token = (
        api.AccessToken(api_key, api_secret)
        .with_identity("agent-anisha")
        .with_name("Anisha")
        .with_grants(api.VideoGrants(room_join=True, room=room_name))
        .to_jwt()
    )

    logger.info(f"Connecting Agent to room: {room_name}")
    await room.connect(url, token)

    # Prewarm VAD
    logger.info("Pre-loading VAD model...")
    vad = silero.VAD.load()

    http_session = aiohttp.ClientSession()
    lkapi = None

    try:
        # Initialize the Agent Session with the required stack
        session = AgentSession(
            stt=deepgram.STT(
                model="nova-3", language="multi", http_session=http_session
            ),
            llm=openai.LLM(
                model="llama-3.3-70b-versatile",
                base_url="https://api.groq.com/openai/v1",
                api_key=os.environ.get("GROQ_API_KEY"),
            ),
            tts=murf.TTS(
                voice="Anisha",
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True,
                http_session=http_session,
            ),
            turn_detection=None,
            vad=vad,
            preemptive_generation=True,
        )

        await session.start(
            agent=Assistant(),
            room=room,
            room_options=room_io.RoomOptions(
                audio_input=room_io.AudioInputOptions(
                    noise_cancellation=lambda params: (
                        noise_cancellation.BVCTelephony()
                        if params.participant.kind
                        == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                        else noise_cancellation.BVC()
                    ),
                ),
            ),
        )

        logger.info("Agent connected and started.")

        @room.on("participant_connected")
        def on_participant_connected(participant: rtc.RemoteParticipant):
            logger.info(
                f"Participant connected: {participant.identity} (kind: {participant.kind})"
            )
            # Inject a hidden/dummy user message to trigger the LLM's outbound opening script
            session.generate_reply(
                user_input="User picked up the phone. Start your opening script now."
            )

        sip_username = my_sip_uri.replace("sip:", "").split("@")[0]
        logger.info(
            f"Initiating SIP call to {sip_username} via trunk {sip_trunk_id}..."
        )
        lkapi = api.LiveKitAPI(url, api_key, api_secret)

        try:
            participant = await lkapi.sip.create_sip_participant(
                api.CreateSIPParticipantRequest(
                    sip_trunk_id=sip_trunk_id,
                    sip_call_to=sip_username,
                    sip_number=sip_username,  # MUST MATCH EXACTLY
                    room_name=room_name,
                    participant_identity="sip-caller",
                    participant_name="Outbound Call",
                )
            )
            logger.info(
                f"SIP call initiated! Participant ID: {participant.participant_id}"
            )
        except Exception as e:
            logger.error(f"Failed to create SIP participant: {e}")

        logger.info("Waiting for the call to finish. Press Ctrl+C to exit.")

        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Exiting script...")
    finally:
        logger.info("Cleaning up...")
        await room.disconnect()
        if lkapi:
            await lkapi.aclose()
        await http_session.close()


if __name__ == "__main__":
    asyncio.run(main())
