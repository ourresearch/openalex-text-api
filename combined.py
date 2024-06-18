from marshmallow import Schema, fields

from concepts import ConceptsSchema
from keywords import KeywordsSchema
from topics import TopicsSchema


class MetaSchema(Schema):
    keywords_count = fields.Int()
    topics_count = fields.Int()
    concepts_count = fields.Int()

    class Meta:
        ordered = True


class CombinedMessageSchema(Schema):
    meta = fields.Nested(MetaSchema)
    keywords = fields.Nested(KeywordsSchema, many=True)
    primary_topic = fields.Nested(TopicsSchema)
    topics = fields.Nested(TopicsSchema, many=True)
    concepts = fields.Nested(ConceptsSchema, many=True)

    class Meta:
        ordered = True
