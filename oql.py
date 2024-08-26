import functools
import json
import os
import random
import requests
import tiktoken
from typing import Union
from flask import jsonify
from openai import OpenAI
from pydantic import BaseModel, StrictStr, StrictBool, StrictFloat, StrictInt
from marshmallow import Schema, fields


# @functools.lru_cache(maxsize=64)
def get_openai_response(prompt):
    api_key = os.getenv("OPENAI_API_KEY")
    # api_key = config_vars['OPENAI_API_KEY']
    client = OpenAI(api_key=api_key)
    oql_entities = get_all_entities_and_columns()

    # Getting examples to feed the model
    messages = example_messages_for_chat(oql_entities)

    # Getting the tools for calling the OpenAlex API
    tools = get_tools()

    # Attaching the new prompt
    messages.append({"role": "user", "content": prompt})

    enc = tiktoken.encoding_for_model("gpt-4o")
    print(len(enc.encode(json.dumps(messages))))

    valid_oql_json_object = False
    i = 0
    # retry the following code 3 times while expression is False
    while not valid_oql_json_object:
        if i == 5:
            break

        # Getting the tool needed for looking up new query
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools
        )
        if response.choices[0].message.tool_calls:
            # Getting institution IDs (if needed)
            institution_ids = use_openai_output_to_get_institution_id(response)

            # Giving data to model to get final json object
            messages.append({"role": "assistant", "content": str(response.choices[0].message)})
            messages.append({"role": "user", "content": json.dumps(institution_ids)})

        completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=messages,
            response_format=OQLJsonObject,
        )
        openai_json_object = json.loads(completion.choices[0].message.content)

        valid_oql_json_object = post_process_openai_output(openai_json_object, oql_entities)
        i += 1

    if not valid_oql_json_object:
        return (jsonify(
                {
                    "error": f"The model is having trouble generating a valid OQL JSON object. Please try again."
                }
            ),
            400,
        )
    else:
        final_json_object = replace_empty_strings_with_none(openai_json_object)
        return final_json_object
    
def replace_empty_strings_with_none(json_object):
    for filter_obj in json_object['filters']:
        if filter_obj['value'] == "":
            filter_obj['value'] = None
            
        if filter_obj['column_id'] == "":
            filter_obj['column_id'] = None
    
    if json_object['summarize_by'] == "":
        json_object['summarize_by'] = None
    return json_object 

def get_institution_id(institution_name: str) -> str:
    # Make a call to the API
    api_call = f"https://api.openalex.org/institutions?search={institution_name}"

    resp = requests.get(api_call)

    if resp.status_code == 200:
        resp_json = resp.json()

        if resp_json['meta']['count'] > 0:
            return resp_json['results'][0]['id'].split("/")[-1].lower()
        else:
            return 'institution not found'
        
    else:
        return 'institution not found'
    
def use_openai_output_to_get_institution_id(chat_response):
    tool_calls = chat_response.choices[0].message.tool_calls

    all_institutions = []
    for tool_call in tool_calls:
        arguments = json.loads(tool_call.function.arguments)
        
        institution_name = arguments.get('institution_name')
    
        institution_id = get_institution_id(institution_name)

        all_institutions.append({"raw_institution_name": institution_name, 
                                 "authorships.institutions.id": f"institutions/{institution_id}", 
                                 "institutions.id": f"institutions/{institution_id}"})

    return all_institutions

def create_system_information(entities_info):
    system_info = "If the name of an institution is given, use the appropriate tool in order to retrieve the OpenAlex institution ID.\n\n"
    system_info = "The value for country or country_code is the 2 letter representation of that country.\n\n"
    system_info = "Filter operator must be one of the following: ['is','is not','is greater than','is less than']\n\n"
    system_info += "Please look at the following subjectEntity information to see which columns can be sorted or filtered or returned and also which ones need to use a function call tool in order to look up the entity:\n\n"
    for entity in entities_info.keys():
        if entities_info[entity]['function_call']:
            system_info += f"subjectEntity: {entity}\n\n"
            system_info += f"Columns (column_id) in {entity} that can be filtered (filters):\n"
            for col in entities_info[entity]['filter']:
                system_info += f"{col}: {entities_info[entity]['columns'][col]}\n"
            system_info += f"\n"
            system_info += f"Columns (column_id) in {entity} that can be sorted (sort_by):\n"
            for col in entities_info[entity]['sort_by']:
                system_info += f"{col}: {entities_info[entity]['columns'][col]}\n"
            system_info += f"\n"
            system_info += f"Columns (column_id) in {entity} that can be returned (return_columns):\n"
            for col in entities_info[entity]['return']:
                system_info += f"{col}: {entities_info[entity]['columns'][col]}\n"
            system_info += f"\n"
            system_info += f"Function call tool needed for {entity}: Yes\n\n\n\n"
        else:
            system_info += f"subjectEntity: {entity}\n\n"
            system_info += f"Columns (column_id) in {entity} that can be filtered (filters):\n"
            for col in entities_info[entity]['filter']:
                system_info += f"{col}: {entities_info[entity]['columns'][col]}\n"
            system_info += f"\n"
            system_info += f"Columns (column_id) in {entity} that can be sorted (sort_by):\n"
            for col in entities_info[entity]['sort_by']:
                system_info += f"{col}: {entities_info[entity]['columns'][col]}\n"
            system_info += f"\n"
            system_info += f"Columns (column_id) in {entity} that can be returned (return_columns):\n"
            for col in entities_info[entity]['return']:
                system_info += f"{col}: {entities_info[entity]['columns'][col]}\n"
            system_info += f"\n"
            system_info += f"Function call tool needed for {entity}: No\n\n"
            system_info += f"Values for {entity}\n"
            for entity_value in entities_info[entity]['values']:
                system_info += f"{entity} value: {entity_value['display_name']}\n{entity} ID: {entity_value['id']}\n\n"
            system_info += f"\n\n\n"
    return system_info.strip()

def example_messages_for_chat(oql_entities):
    first_example = "Just list all of the works in OpenAlex"

    first_example_answer = {"filters": [{"id": "branch_1",
                                        "subjectEntity": "works",
                                        "type": "branch",
                                        "column_id": "",
                                        "operator": "and",
                                        "value": "",
                                        "children": []}],
                            "summarize_by": "",
                            "sort_by": {},
                            "return_columns": []}
    second_example = "List all works from North Carolina State University (using the OpenAlex ID) in 2023 and show me the openalex ID, title, and publication year. Sort by publication year with the newest publications first."
    
    information_for_system = create_system_information(oql_entities)

    messages = [
        {"role": "system", 
         "content": "You are helping to take in database search requests from users and turn them into a JSON object."},
        {"role": "user", "content": information_for_system},
        {"role": "assistant", 
         "content": "I will refer back to this information when determining which columns need to be filtered, sorted, or returned"},
        {"role": "user","content": first_example}, 
        {"role": "user","content": json.dumps(first_example_answer)}, 
        {"role": "user","content": second_example}]

    messages.append({"role": "assistant", "content": """ChatCompletionMessage(content=None, refusal=None, role='assistant', function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_fcEKw4AeBTklT7HtJyakgboc', function=Function(arguments='{"institution_name":"North Carolina State University"}', name='get_institution_id'), type='function')])"""})
    messages.append({"role": "user", "content": json.dumps([{'raw_institution_name': 'North Carolina State University', 
                                                             'authorships.institutions.id': 'institutions/i137902535', 
                                                             'institutions.id': 'institutions/i137902535'}])})
    messages.append({"role": "assistant", "content": json.dumps({"filters": 
                                                                        [{
                                                                            "id": "branch_1",
                                                                            "subjectEntity": "works",
                                                                            "type": "branch",
                                                                            "column_id": "",
                                                                            "operator": "and",
                                                                            "value": "",
                                                                            "children": [
                                                                                "leaf_1",
                                                                                "leaf_2"]},
                                                                         {"id": "leaf_1",
                                                                          "subjectEntity": "works",
                                                                          "type": "leaf",
                                                                          "column_id": "authorships.institutions.id",
                                                                          "operator": "is",
                                                                          "value": "institutions/I137902535", 
                                                                          "children": []},
                                                                          {"id": "leaf_2",
                                                                          "subjectEntity": "works",
                                                                          "type": "leaf",
                                                                          "column_id": "authorships.institutions.id",
                                                                          "operator": "is",
                                                                          "value": "institutions/I137902535", 
                                                                          "children": []}
                                                                        ],
                                                                        "summarize_by": "",
                                                                        "sort_by": {
                                                                            "column_id": "publication_year",
                                                                            "direction": "desc"
                                                                            },
                                                                        "return_columns": [
                                                                            "openalex_id",
                                                                            "paper_title",
                                                                            "publication_year"
                                                                            ]})})
    messages.append({"role": "user", "content": "Give me high level information about French institutions"})
    messages.append({"role": "assistant", "content": json.dumps({"filters": 
                                                                        [{
                                                                            "id": "branch_1",
                                                                            "subjectEntity": "works",
                                                                            "type": "branch",
                                                                            "column_id": "",
                                                                            "operator": "and",
                                                                            "value": "",
                                                                            "children": [
                                                                                "leaf_1"
                                                                            ]
                                                                        },
                                                                         {
                                                                            "id": "leaf_1",
                                                                            "subjectEntity": "works",
                                                                            "type": "leaf",
                                                                            "column_id": "authorships.countries",
                                                                            "operator": "is",
                                                                            "value": "countries/FR",
                                                                            "children": []
                                                                            }],
                                                                        "summarize_by": "institutions",
                                                                        "sort_by": {
                                                                            "column_id": "count",
                                                                            "direction": "desc"
                                                                            },
                                                                        "return_columns": [
                                                                            "id",
                                                                            "display_name",
                                                                            "ids.ror",
                                                                            "type",
                                                                            "mean(fwci)",
                                                                            "count"
                                                                            ]})})
    
    messages.append({"role": "user", "content": json.dumps("I want to see all works from Sorbonne University that are open access and in English while also being tagged with the SDG for good health and well-being.")})
    messages.append({"role": "assistant", "content": json.dumps({"filters": [
        {
            "id": "br_ikDytJ",
            "subjectEntity": "works",
            "type": "branch",
            "operator": "and",
            "children": [
                "leaf_9Uc8qP",
                "leaf_5yAEm5",
                "leaf_3aJfuC",
                "leaf_971QRL"
            ]
        },
        {
            "id": "leaf_9Uc8qP",
            "subjectEntity": "works",
            "type": "leaf",
            "column_id": "authorships.institutions.id",
            "operator": "is",
            "value": "institutions/I39804081"
        },
        {
            "id": "leaf_5yAEm5",
            "subjectEntity": "works",
            "type": "leaf",
            "column_id": "open_access.is_oa",
            "operator": "is",
            "value": True
        },
        {
            "id": "leaf_3aJfuC",
            "subjectEntity": "works",
            "type": "leaf",
            "column_id": "language",
            "operator": "is",
            "value": "languages/en"
        },
        {
            "id": "leaf_971QRL",
            "subjectEntity": "works",
            "type": "leaf",
            "column_id": "sustainable_development_goals.id",
            "operator": "is",
            "value": "sdgs/3"
        }
    ],
    "summarize_by": "",
    "sort_by": {
    "column_id": "publication_year",
    "direction": "desc"
    },
    "return_columns": [
        "display_name",
        "publication_year",
        "type",
        "cited_by_count"
        ]
    })})

    return messages

def get_tools():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_institution_id",
                "description": "Get the OpenAlex institution ID from the API when a institution name needs to be looked up to find the ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "institution_name": {
                            "type": "string",
                            "description": "The name of the institution that needs to be looked up.",
                        },
                    },
                    "required": ["institution_name"],
                    "additionalProperties": False,
                },
            }
        }
    ]
    return tools

def get_all_entities_and_columns():
    entities_with_function_calling = ['works','institutions']
    entities_with_function_calling_not_set_up = ['authors','sources','topics','concepts','funders','keywords','publishers']
    entities_without_function_calling = ['continents', 'countries', 'domains','fields','institution-types','languages','licenses',
                                         'sdgs','source-types','subfields','types']
    
    config_json = requests.get("https://api.openalex.org/entities/config").json()

    oql_info = {}
    for key in config_json.keys():
        if key in entities_with_function_calling:
            oql_info[key] = {'function_call': True, 
                             'columns': {}, 
                             'values': {}, 
                             'descr': config_json[key]['descrFull']} # put descr into 'columns' later
            cols_for_filter = []
            cols_for_sort_by = []
            cols_for_return = []
            for col in config_json[key]['columns'].keys():
                oql_info[key]['columns'][config_json[key]['columns'][col]['id']] = config_json[key]['columns'][col]['descr']
                # if 'filter' in entity_configs[key]['columns'][col]['actions']:
                #     cols_for_filter.append(entity_configs[key]['columns'][col]['id'])
                cols_for_filter.append(config_json[key]['columns'][col]['id'])

                if config_json[key]['columns'][col].get('actions') and ('sort' in config_json[key]['columns'][col]['actions']):
                    cols_for_sort_by.append(config_json[key]['columns'][col]['id'])

                cols_for_return.append(config_json[key]['columns'][col]['id'])
            oql_info[key]['filter'] = cols_for_filter
            oql_info[key]['sort_by'] = cols_for_sort_by
            oql_info[key]['return'] = cols_for_return
        elif key in entities_without_function_calling:
            oql_info[key] = {'function_call': False, 
                             'columns': {}, 
                             'values': config_json[key]['values'],
                             'descr': config_json[key]['descrFull']} # put descr into 'columns' later
            cols_for_filter = []
            cols_for_sort_by = []
            cols_for_return = []
            for col in config_json[key]['columns'].keys():
                oql_info[key]['columns'][config_json[key]['columns'][col]['id']] = config_json[key]['columns'][col]['descr']
                # if 'filter' in entity_configs[key]['columns'][col]['actions']:
                #     cols_for_filter.append(entity_configs[key]['columns'][col]['id'])
                cols_for_filter.append(config_json[key]['columns'][col]['id'])

                if config_json[key]['columns'][col].get('actions') and ('sort' in config_json[key]['columns'][col]['actions']):
                    cols_for_sort_by.append(config_json[key]['columns'][col]['id'])

                cols_for_return.append(config_json[key]['columns'][col]['id'])
                
            oql_info[key]['filter'] = cols_for_filter
            oql_info[key]['sort_by'] = cols_for_sort_by
            oql_info[key]['return'] = cols_for_return
            

    return oql_info

def check_filters(all_filters, oql_entity_info):
    for one_filter in all_filters:
        if one_filter['subjectEntity'] not in oql_entity_info.keys():
            return False
        else:
            if one_filter['column_id'] not in oql_entity_info[one_filter['subjectEntity']]['filter']:
                if one_filter['column_id'] == "":
                    return True 
                else:
                    return False
            else:
                return True
    return True

def check_summarize_by(summarize_by, oql_entity_info):
    if summarize_by:
        if summarize_by in oql_entity_info.keys():
            return True
        elif summarize_by == "all":
            return True
        elif summarize_by == "":
            return True
        else:
            return False
    else:
        return True

def check_sort_by(sort_by, oql_entity_info, summarize_by):
    if summarize_by:
        entity_sorting_cols = oql_entity_info[summarize_by]['sort_by']
    else:
        entity_sorting_cols = oql_entity_info["works"]['sort_by']

    if sort_by:
        if sort_by['column_id'] in entity_sorting_cols:
            if sort_by['direction'] in ['asc', 'desc']:
                return True
            else:
                return False
        else:
            return False
    else:
        return True

def check_return_columns(return_cols, oql_entity_info, summarize_by):
    if summarize_by:
        entity_return_cols = oql_entity_info[summarize_by]['return']
    else:
        entity_return_cols = oql_entity_info["works"]['return']

    if return_cols:
        if all(col in entity_return_cols for col in return_cols):
            return True
        else:
            return False
    else:
        return True

def post_process_openai_output(openai_dict, oql_entities):
    print(openai_dict)
    if all(key in list(openai_dict.keys()) for key in ['filters', 'summarize_by', 'sort_by', 'return_columns']):
        if check_filters(openai_dict['filters'], oql_entities):
            if check_summarize_by(openai_dict['summarize_by'], oql_entities):
                if check_sort_by(openai_dict['sort_by'], oql_entities, openai_dict['summarize_by']):
                    if check_return_columns(openai_dict['return_columns'], oql_entities, openai_dict['summarize_by']):
                        return True
                    else:
                        return False
                else:
                    return False
            else:
                return False
        else:
            return False
    else:
        return False

class FilterObject(BaseModel):
    id: str
    subjectEntity: str
    type: str
    column_id: str = None
    operator: str
    value: Union[StrictStr, StrictBool, StrictFloat, StrictInt] = None
    children: list[str] = None

# class BranchFilterObject(BaseModel):
#     subjectEntity: str
#     id: str
#     type: str
#     operator: str
#     children: list[str]

class SortByObject(BaseModel):
    column_id: str
    direction: str
    
class OQLJsonObject(BaseModel):
    filters: list[FilterObject]
    summarize_by: str
    sort_by: SortByObject
    return_columns: list[str]

# class LeafFiltersSchema(Schema):
#     id = fields.Str()
#     subjectEntity = fields.Str()
#     type = fields.Str()
#     column_id = fields.Str()
#     operator = fields.Str()
#     value = fields.Raw()

#     class Meta:
#         ordered = True
# class BranchFiltersSchema(Schema):
#     id = fields.Str()
#     subjectEntity = fields.Str()
#     type = fields.Str()
#     operator = fields.Str()
#     children = fields.Nested(fields.Str(), many=True)
#     value = fields.Raw()

#     class Meta:
#         ordered = True

# class SortBySchema(Schema):
#     column_id = fields.Str()
#     direction = fields.Str()

#     class Meta:
#         ordered = True

# class ValueField(fields.Field):
#     def _deserialize(self, value, attr, data, **kwargs):
#         if 'children' in value.keys():
#             if BranchFiltersSchema().load(value):
#                 return 'yes'
#             else:
#                 return 'no'
#         elif 'value' in value.keys():
#             if LeafFiltersSchema().load(value):
#                 return 'yes'
#             else:
#                 return 'no'
#         else:
#             raise marshmallow.expections.ValidationError('Schema should be branch or leaf')

# class JsonObjectSchema(Schema):
#     filters = ValueField()
#     summarize_by = fields.Str(nullable=True)
#     sort_by = fields.Nested(SortBySchema)
#     return_columns = fields.List(fields.Str())

#     class Meta:
#         ordered = True
