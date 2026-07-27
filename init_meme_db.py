from qdrant_client.models import Distance, PayloadSchemaType, TextIndexParams, TokenizerType, VectorParams

from config import MEME_COLLECTION_NAME, qdrant_client


def initialize_meme_database() -> None:
    client = qdrant_client()
    if not client.collection_exists(MEME_COLLECTION_NAME):
        client.create_collection(
            collection_name=MEME_COLLECTION_NAME,
            vectors_config={
                "visual": VectorParams(size=512, distance=Distance.COSINE),
                "semantic_text": VectorParams(size=384, distance=Distance.COSINE),
            },
        )
    # Payload indexes make template/tag filtering fast. Text indexing is
    # best-effort because older Qdrant versions may not expose all options.
    for field, schema in (("template", PayloadSchemaType.KEYWORD), ("tags", PayloadSchemaType.KEYWORD)):
        try:
            client.create_payload_index(collection_name=MEME_COLLECTION_NAME, field_name=field, field_schema=schema)
        except Exception:
            pass
    try:
        client.create_payload_index(
            collection_name=MEME_COLLECTION_NAME,
            field_name="normalized_text",
            field_schema=TextIndexParams(type="text", tokenizer=TokenizerType.WORD, lowercase=True),
        )
    except Exception:
        pass
    print(f"Meme collection ready: {MEME_COLLECTION_NAME}")


if __name__ == "__main__":
    initialize_meme_database()
