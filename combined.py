from marshmallow import Schema, fields

from concepts import ConceptsSchema
from topics import TopicsSchema


class MetaSchema(Schema):
    concepts_count = fields.Int()
    topics_count = fields.Int()

    class Meta:
        ordered = True


class CombinedMessageSchema(Schema):
    meta = fields.Nested(MetaSchema)
    concepts = fields.Nested(ConceptsSchema, many=True)
    topics = fields.Nested(TopicsSchema, many=True)

    class Meta:
        ordered = True
