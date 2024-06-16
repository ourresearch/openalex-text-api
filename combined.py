from marshmallow import Schema, fields

from concepts import ConceptsSchema
from topics import TopicsSchema


class MetaSchema(Schema):
    topics_count = fields.Int()
    concepts_count = fields.Int()

    class Meta:
        ordered = True


class CombinedMessageSchema(Schema):
    meta = fields.Nested(MetaSchema)
    topics = fields.Nested(TopicsSchema, many=True)
    concepts = fields.Nested(ConceptsSchema, many=True)

    class Meta:
        ordered = True
