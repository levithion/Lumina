from qdrant_client.models import Distance, PayloadSchemaType, TextIndexParams, TokenizerType, VectorParams

from config import MEME_COLLECTION_NAME, qdrant_client


def initialize_meme_database(collection_name: str = "") -> None:
    client = qdrant_client()
    name = collection_name or MEME_COLLECTION_NAME
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config={
                "visual": VectorParams(size=512, distance=Distance.COSINE),
                "semantic_text": VectorParams(size=384, distance=Distance.COSINE),
            },
        )
    # Payload indexes make filter pushdown fast. template_key backs the
    # case-insensitive template filter; is_sensitive backs safe-mode filtering.
    for field, schema in (
        ("template", PayloadSchemaType.KEYWORD),
        ("template_key", PayloadSchemaType.KEYWORD),
        ("tags", PayloadSchemaType.KEYWORD),
        ("is_sensitive", PayloadSchemaType.BOOL),
    ):
        try:
            client.create_payload_index(collection_name=name, field_name=field, field_schema=schema)
        except Exception:
            pass
    try:
        client.create_payload_index(
            collection_name=name,
            field_name="normalized_text",
            field_schema=TextIndexParams(type="text", tokenizer=TokenizerType.WORD, lowercase=True),
        )
    except Exception:
        pass
    print(f"Meme collection ready: {name}")


if __name__ == "__main__":
    initialize_meme_database()
