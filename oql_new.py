import functools
import json
import os
import random
import requests
# import tiktoken
from typing import Union
from flask import jsonify
from openai import OpenAI
from pydantic import BaseModel, StrictStr, StrictBool, StrictFloat, StrictInt
from marshmallow import Schema, fields
from oqo_validate import OQOValidator

# @functools.lru_cache(maxsize=64)
def get_openai_response(prompt):
    api_key = os.getenv("OPENAI_API_KEY")
    # api_key = config_vars['OPENAI_API_KEY']
    client = OpenAI(api_key=api_key)
    oql_entities = get_all_entities_and_columns()

    # Prompt safety check
    prompt_ok = check_prompt_for_safety(prompt)

    # Load the OQO validator
    validator = OQOValidator()
    
    if not prompt_ok:
        return (jsonify(
            {
                "error": f"The prompt did not pass the initial check. Please try again."
            }
        ),
        400,
        )

    # Figuring out which parts need to be figured out by the model
    messages_parsed = messages_for_parse_prompt(oql_entities)
    messages_parsed.append({"role": "user", "content": prompt})

    completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=messages_parsed,
            response_format=ParsedPromptObject,
        )
    
    parsed_prompt = json.loads(completion.choices[0].message.content)

    # Getting examples to feed the model
    messages = example_messages_for_chat(oql_entities)

    if (not parsed_prompt['filters_needed'] and 
        not parsed_prompt['summarize_by_filters_needed'] and
        not parsed_prompt['sort_by_needed'] and 
        not parsed_prompt['return_columns_needed']):
        if parsed_prompt['summarize_by'] == "":
            return {}
        else:
            json_object = {"summarize_by": parsed_prompt['summarize_by']}
            ok, error_message = validator.validate(json_object)
            if ok:
                return json_object
            else:
                return (
                    jsonify(
                        {
                            "error": error_message
                            }
                            ),
                            400,
                            )
    
    if (not parsed_prompt['filters_needed'] and 
        not parsed_prompt['summarize_by_filters_needed'] and
        not parsed_prompt['sort_by_needed'] and 
        parsed_prompt['return_columns_needed']):

        messages.append({"role": "user", "content": f"Give list of return columns for this text: {prompt}"})

        completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=messages,
            response_format=ReturnColumnsObject,
        )

        json_object = json.loads(completion.choices[0].message.content)
        ok, error_message = validator.validate(json_object)

        if ok:
            return json_object
        else:
            return (
                jsonify(
                    {
                        "error": error_message
                        }
                        ),
                        400,
                        )

    elif (not parsed_prompt['filters_needed'] and 
        not parsed_prompt['summarize_by_filters_needed'] and
        parsed_prompt['sort_by_needed'] and 
        not parsed_prompt['return_columns_needed']):
        messages.append({"role": "user", "content": f"Give the sort by columns for this text: {prompt}"})

        completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=messages,
            response_format=SortByColumnsObject,
        )

        json_object = json.loads(completion.choices[0].message.content)
        ok, error_message = validator.validate(json_object)

        if ok:
            return json_object
        else:
            return (
                jsonify(
                    {
                        "error": error_message
                        }
                        ),
                        400,
                        )
    
    elif (not parsed_prompt['filters_needed'] and 
        not parsed_prompt['summarize_by_filters_needed'] and
        parsed_prompt['sort_by_needed'] and 
        parsed_prompt['return_columns_needed']):

        messages.append({"role": "user", "content": f"Give the sort by and return columns for this text: {prompt}"})

        completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=messages,
            response_format=ReturnSortByColumnsObject,
        )

        json_object = json.loads(completion.choices[0].message.content)
        ok, error_message = validator.validate(json_object)

        if ok:
            return json_object
        else:
            return (
                jsonify(
                    {
                        "error": error_message
                        }
                        ),
                        400,
                        )

    # Getting the tools for calling the OpenAlex API
    tools = get_tools()

    # Attaching the new prompt
    messages.append({"role": "user", "content": prompt})

    # enc = tiktoken.encoding_for_model("gpt-4o")
    # print(len(enc.encode(json.dumps(messages))))

    ok = False
    validator = OQOValidator()
    i = 0
    # retry the following code 5 times while expression is False
    while not ok:
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
            all_ids = use_openai_output_to_get_ids(response)

            # Giving data to model to get final json object
            messages.append({"role": "assistant", "content": str(response.choices[0].message)})
            messages.append({"role": "user", "content": json.dumps(all_ids)})

        completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=messages,
            response_format=OQLJsonObject,
        )
        openai_json_object = json.loads(completion.choices[0].message.content)
        print(openai_json_object)
        print("")
        ok, error_message = validator.validate(openai_json_object)
        messages.append({"role": "assistant", "content": str(completion.choices[0].message.content)})
        messages.append({"role": "user", "content": f"That was not correct. The following error message was received:\n{error_message}\n\nPlease try again."})
        # valid_oql_json_object = post_process_openai_output(openai_json_object, oql_entities)
        i += 1

    if not ok:
        return (jsonify(
                {
                    "error": f"The model is having trouble generating a valid OQL JSON object. The latest error message received was '{error_message}'. Please try again."
                }
            ),
            400,
        )
    else:
        final_json_object = fix_output_for_final(openai_json_object)

        if not parsed_prompt['sort_by_needed']:
            _ = final_json_object.pop('sort_by')
        if not parsed_prompt['return_columns_needed']:
            _ = final_json_object.pop('return_columns')
        if not final_json_object['summarize_by']:
            _ = final_json_object.pop('summarize_by')
        
        final_val, final_error  = validator.validate(final_json_object)
        if final_val:
            return final_json_object
        else:
           return (jsonify(
                {
                    "error": f"The model is having trouble generating a the final OQL JSON object. Getting the following message: {final_error}"
                }
            ),
            400,
            ) 
        
def get_openai_response_for_parsing_prompt(prompt, api_key, openai_client):
    return None

def create_filter_objects(prompt, api_key, openai_client):
    return None

def check_prompt_for_safety(prompt):
    if len(prompt) > 1000:
        return False
    else:
        return True

def messages_for_parse_prompt(oql_entities):
    information_for_system = create_system_information(oql_entities)

    example_1 = "Show me all works in OpenAlex"
    example_1_answer = {
        "filters_needed": False,
        "summarize_by": "",
        "summarize_by_filters_needed": False,
        "sort_by_needed": False,
        "return_columns_needed": False
        }
    
    example_2 = "Show me all works from North Carolina State University in 2023 and show me the openalex ID, title, and cited by count. Show the highest cited publications first."
    example_2_answer = {
        "filters_needed": True,
        "summarize_by": "",
        "summarize_by_filters_needed": False,
        "sort_by_needed": True,
        "return_columns_needed": True
        }
    
    example_3 = "Which institutions does NASA collaborate the most with in Africa?"
    example_3_answer = {
        "filters_needed": True,
        "summarize_by": "institutions",
        "summarize_by_filters_needed": True,
        "sort_by_needed": True,
        "return_columns_needed": False
        }
    
    example_4 = "Which researchers at the University of Colorado have published the most work on SDG 13?"
    example_4_answer = {
        "filters_needed": True,
        "summarize_by": "authors",
        "summarize_by_filters_needed": True,
        "sort_by_needed": True,
        "return_columns_needed": False
        }
    
    example_5 = "Which journals publish the highest cited research on coral bleaching??"
    example_5_answer = {
        "filters_needed": True,
        "summarize_by": "sources",
        "summarize_by_filters_needed": True,
        "sort_by_needed": True,
        "return_columns_needed": False
        }
    
    example_6 = "authors or show me all authors or get authors"
    example_6_answer = {
        "filters_needed": False,
        "summarize_by": "authors",
        "summarize_by_filters_needed": False,
        "sort_by_needed": False,
        "return_columns_needed": False
        }

    messages = [
        {"role": "system", 
         "content": "You are helping to take in database search requests from users and parse them into different parts."},
        {"role": "user", "content": information_for_system},
        {"role": "assistant", 
         "content": "I will refer back to this information when determining the different elements of the prompt"},
        {"role": "user","content": example_1}, 
        {"role": "user","content": json.dumps(example_1_answer)}, 
        {"role": "user","content": example_2}, 
        {"role": "user","content": json.dumps(example_2_answer)}, 
        {"role": "user","content": example_3}, 
        {"role": "user","content": json.dumps(example_3_answer)},
        {"role": "user","content": example_4}, 
        {"role": "user","content": json.dumps(example_4_answer)},
        {"role": "user","content": example_5}, 
        {"role": "user","content": json.dumps(example_5_answer)},
        {"role": "user","content": example_6}, 
        {"role": "user","content": json.dumps(example_6_answer)}
    ]
    return messages
    
def fix_output_for_final(json_object):
    for filter_obj in json_object['filters']:
        if filter_obj['value'] == "":
            filter_obj['value'] = None
            
        if filter_obj['column_id'] == "":
            filter_obj['column_id'] = None
    
    if json_object['summarize_by'] == "":
        json_object['summarize_by'] = None
    elif json_object['summarize_by'] == "works":
        json_object['summarize_by'] = "all"

    final_filter_obj = []
    branch_objs = 0
    for filter_obj in json_object['filters']:
        if filter_obj['type'] == "branch":
            if branch_objs == 0:
                if (filter_obj['subjectEntity'] != "works") and not json_object['summarize_by']:
                    filter_obj['subjectEntity'] = "works"
            branch_objs += 1
            final_filter_obj.append({k: v for k, v in filter_obj.items() if k not in ['value','column_id']})
        elif filter_obj['type'] == "leaf":
            if filter_obj['operator'] in ['>', '<', '>=','<=']:
                filter_obj['operator'] = f"is {filter_obj['operator'].replace('>=', 'greater than or equal to').replace('<=', 'less than or equal to').replace('>', 'greater than').replace('<', 'less than')}"
            if isinstance(filter_obj['value'], str):
                if 'works' in filter_obj['value']:
                    if 'works/W' in filter_obj['value']:
                        filter_obj['value'] = filter_obj['value'].split("works/W")[1]
            final_filter_obj.append({k: v for k, v in filter_obj.items() if k not in ['children']})

    # check if final_filter_obj is leaf only
    if (branch_objs == 0) and (len(final_filter_obj) >= 1):
        new_final_filter_obj = [
            {
                "id": "branch_work",
                "subjectEntity": "works",
                "type": "branch",
                "operator": "and",
                "children": [x['id'] for x in final_filter_obj if x['subjectEntity'] == "works"]
            }
        ]
        _ = [new_final_filter_obj.append(x) for x in final_filter_obj]
        final_filter_obj = new_final_filter_obj

    json_object['filters'] = final_filter_obj
    return json_object 

def get_institution_id(institution_name: str) -> str:
    # Make a call to the API
    api_call = f"https://api.openalex.org/institutions?search={institution_name}"

    resp = requests.get(api_call)

    if resp.status_code == 200:
        resp_json = resp.json()

        if resp_json['meta']['count'] > 0:
            return resp_json['results'][0]['id'].split("/")[-1]
        else:
            return 'institution not found'
        
    else:
        return 'institution not found'
    
def get_author_id(author_name: str) -> str:
    # Make a call to the API
    api_call = f"https://api.openalex.org/authors?search={author_name}"

    resp = requests.get(api_call)

    if resp.status_code == 200:
        resp_json = resp.json()

        if resp_json['meta']['count'] > 0:
            return resp_json['results'][0]['id'].split("/")[-1]
        else:
            return 'author not found'
        
    else:
        return 'author not found'
    
def get_keyword_id(keyword_name: str) -> str:
    # Make a call to the API
    api_call = f"https://api.openalex.org/keywords?search={keyword_name}"

    resp = requests.get(api_call)

    if resp.status_code == 200:
        resp_json = resp.json()

        if resp_json['meta']['count'] > 0:
            return resp_json['results'][0]['id'].split("/")[-1].lower()
        else:
            return 'keyword not found'
        
    else:
        return 'keyword not found'
    
def get_source_id(source_name: str) -> str:
    # Make a call to the API
    api_call = f"https://api.openalex.org/sources?search={source_name}"

    resp = requests.get(api_call)

    if resp.status_code == 200:
        resp_json = resp.json()

        if resp_json['meta']['count'] > 0:
            return resp_json['results'][0]['id'].split("/")[-1].lower()
        else:
            return 'source not found'
        
    else:
        return 'source not found'
    
def get_funder_id(funder_name: str) -> str:
    # Make a call to the API
    api_call = f"https://api.openalex.org/funders?search={funder_name}"

    resp = requests.get(api_call)

    if resp.status_code == 200:
        resp_json = resp.json()

        if resp_json['meta']['count'] > 0:
            return resp_json['results'][0]['id'].split("/")[-1].lower()
        else:
            return 'funder not found'
        
    else:
        return 'funder not found'

def get_publisher_id(publisher_name: str) -> str:
    # Make a call to the API
    api_call = f"https://api.openalex.org/publishers?search={publisher_name}"

    resp = requests.get(api_call)

    if resp.status_code == 200:
        resp_json = resp.json()

        if resp_json['meta']['count'] > 0:
            return resp_json['results'][0]['id'].split("/")[-1].lower()
        else:
            return 'publisher not found'
        
    else:
        return 'publisher not found'
    
def get_topic_id(topic_name: str) -> str:
    # Make a call to the API
    api_call = f"https://api.openalex.org/topics?search={topic_name}"

    resp = requests.get(api_call)

    if resp.status_code == 200:
        resp_json = resp.json()

        if resp_json['meta']['count'] > 0:
            return resp_json['results'][0]['id'].split("/")[-1].lower()
        else:
            return 'topic not found'
        
    else:
        return 'topic not found'
    
def use_openai_output_to_get_ids(chat_response):
    tool_calls = chat_response.choices[0].message.tool_calls

    all_tool_data = []
    for tool_call in tool_calls:
        arguments = json.loads(tool_call.function.arguments)
        
        if tool_call.function.name == "get_institution_id":
            institution_name = arguments.get('institution_name')
        
            institution_id = get_institution_id(institution_name)

            all_tool_data.append({"raw_institution_name": institution_name, 
                                    "authorships.institutions.id": f"institutions/{institution_id}", 
                                    "institutions.id": f"institutions/{institution_id}"})
        elif tool_call.function.name == "get_author_id":
            author_name = arguments.get('author_name')
        
            author_id = get_author_id(author_name)

            all_tool_data.append({"raw_author_name": author_name, 
                                    "authorships.authors.id": f"authors/{author_id}", 
                                    "authors.id": f"authors/{author_id}"})
        elif tool_call.function.name == "get_keyword_id":
            search_name = arguments.get('search_name')
        
            keyword_id = get_keyword_id(search_name)

            all_tool_data.append({"raw_search_name": search_name, 
                                  "keywords.id": f"keywords/{keyword_id}"})
            
        elif tool_call.function.name == "get_source_id":
            search_name = arguments.get('search_name')
        
            source_id = get_source_id(search_name)

            all_tool_data.append({"raw_search_name": search_name, 
                                  "primary_location.source.id": f"sources/{source_id}"})
            
        elif tool_call.function.name == "get_funder_id":
            search_name = arguments.get('search_name')
        
            funder_id = get_funder_id(search_name)

            all_tool_data.append({"raw_search_name": search_name, 
                                  "grants.funder": f"funders/{funder_id}"})
            
        elif tool_call.function.name == "get_publisher_id":
            search_name = arguments.get('search_name')
        
            publisher_id = get_publisher_id(search_name)

            all_tool_data.append({"raw_search_name": search_name, 
                                  "primary_location.source.publisher_lineage": f"publishers/{publisher_id}"})
            
        elif tool_call.function.name == "get_topic_id":
            search_name = arguments.get('search_name')
        
            topic_id = get_topic_id(search_name)

            all_tool_data.append({"raw_search_name": search_name, 
                                  "primary_topic.id": f"topics/{topic_id}"})
    return all_tool_data

def create_system_information(entities_info):
    system_info = "If the name of an institution is given, use the appropriate tool in order to retrieve the OpenAlex institution ID.\n\n"
    system_info = "The value for country or country_code is the 2 letter representation of that country.\n\n"
    system_info = "Filter operator must be one of the following: ['is','is not','is greater than','is less than']\n\n"
    system_info = "Default to sorting by 'cited_by_count' if possible unless another sorting column_id is specified by the user.\n\n"
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
    information_for_system = create_system_information(oql_entities)

    example_1 = "Just list all of the works in OpenAlex (also known as 'get works')"
    example_1_answer = json.dumps({
        "filters": [
            {
                "id": "branch_work",
                "subjectEntity": "works",
                "type": "branch",
                "column_id": "",
                "operator": "and",
                "value": "",
                "children": []
            }
        ],
        "summarize_by": "",
        "sort_by": {
            "column_id": "cited_by_count",
            "direction": "desc"
        },
        "return_columns": []})
    
    example_1 = "What respositories are indexed in OpenAlex?"
    example_1_answer = json.dumps({
        "filters": [
            {
                "id": "branch_work",
                "subjectEntity": "works",
                "type": "branch",
                "column_id": "",
                "operator": "and",
                "value": "",
                "children": []
            },
            {
                "id": "branch_source",
                "subjectEntity": "source",
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
                "subjectEntity": "source",
                "type": "leaf",
                "column_id": "source_type",
                "operator": "is",
                "value": "source-types/repositories",
                "children": []
            }
        ],
        "summarize_by": "sources",
        "sort_by": {
            "column_id": "count",
            "direction": "desc"
        },
        "return_columns": [
            "display_name",
            "count"
        ]})
    
    example_2 = "List all works from North Carolina State University (using the OpenAlex ID) in 2023 and show me the openalex ID, title, and cited by count. Show the highest cited publications first."
    example_2_tool = """ChatCompletionMessage(content=None, refusal=None, role='assistant', function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_fcEKw4AeBTklT7HtJyakgboc', function=Function(arguments='{"institution_name":"North Carolina State University"}', name='get_institution_id'), type='function')])"""
    example_2_tool_response = json.dumps([{'raw_institution_name': 'North Carolina State University', 
                                           'authorships.institutions.id': 'institutions/I137902535', 
                                           'institutions.id': 'institutions/I137902535'}])
    example_2_answer = json.dumps({
        "filters": 
        [{
            "id": "branch_work",
            "subjectEntity": "works",
            "type": "branch",
            "column_id": "",
            "operator": "and",
            "value": "",
            "children": [
                "leaf_1",
                "leaf_2"
            ]
        },
        {
            "id": "leaf_1",
            "subjectEntity": "works",
            "type": "leaf",
            "column_id": "publication_year",
            "operator": "is",
            "value": "2023", 
            "children": []
        },
        {
            "id": "leaf_2",
            "subjectEntity": "works",
            "type": "leaf",
            "column_id": "authorships.institutions.id",
            "operator": "is",
            "value": "institutions/I137902535", 
            "children": []
        }],
        "summarize_by": "",
        "sort_by": {
            "column_id": "cited_by_count",
            "direction": "desc"
        },
        "return_columns": [
            "openalex_id",
            "paper_title",
            "cited_by_count"
        ]})
    

    example_3 = "Give me high level information for French institutions (summarize)"
    example_3_answer = json.dumps(
        {"filters": 
        [{
            "id": "branch_work",
            "subjectEntity": "works",
            "type": "branch",
            "column_id": "",
            "operator": "and",
            "value": "",
            "children": []
        },
        {
            "id": "branch_institution",
            "subjectEntity": "institutions",
            "type": "branch",
            "column_id": "",
            "operator": "and",
            "value": "",
            "children": ["leaf_1"]
        },
        {
            "id": "leaf_1",
            "subjectEntity": "works",
            "type": "leaf",
            "column_id": "authorships.countries",
            "operator": "is",
            "value": "countries/FR",
            "children": []
        },
        ],
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
            ]})
    
    example_4 = "I want to see all works from Sorbonne University that are open access and in English while also being tagged with the SDG for good health and well-being."
    example_4_tool = """ChatCompletionMessage(content=None, refusal=None, role='assistant', function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_DOSHfhsdiSFhsFHsAH', function=Function(arguments='{"institution_name":"Sorbonne University"}', name='get_institution_id'), type='function')])"""
    example_4_tool_response = json.dumps([{'raw_institution_name': 'Sorbonne University', 
                                           'authorships.institutions.id': 'institutions/I39804081', 
                                           'institutions.id': 'institutions/I39804081'}])
    example_4_answer = json.dumps({
        "filters": [
        {
            "id": "branch_work",
            "subjectEntity": "works",
            "type": "branch",
            "operator": "and",
            "children": [
                "leaf_1",
                "leaf_2",
                "leaf_3",
                "leaf_4"
            ]
        },
        {
            "id": "leaf_1",
            "subjectEntity": "works",
            "type": "leaf",
            "column_id": "authorships.institutions.id",
            "operator": "is",
            "value": "institutions/I39804081"
        },
        {
            "id": "leaf_2",
            "subjectEntity": "works",
            "type": "leaf",
            "column_id": "open_access.is_oa",
            "operator": "is",
            "value": True
        },
        {
            "id": "leaf_3",
            "subjectEntity": "works",
            "type": "leaf",
            "column_id": "language",
            "operator": "is",
            "value": "languages/en"
        },
        {
            "id": "leaf_4",
            "subjectEntity": "works",
            "type": "leaf",
            "column_id": "sustainable_development_goals.id",
            "operator": "is",
            "value": "sdgs/3"
        }
    ],
    "summarize_by": "",
    "sort_by": {
    "column_id": "cited_by_count",
    "direction": "desc"
    },
    "return_columns": [
        "display_name",
        "publication_year",
        "type",
        "cited_by_count"
        ]
    })

    example_5 = "Show me African institutions that collaborate with MIT the most."
    example_5_tool = """ChatCompletionMessage(content=None, refusal=None, role='assistant', function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_DOSHfhsdiSFhsFHsAH', function=Function(arguments='{"institution_name":"MIT"}', name='get_institution_id'), type='function')])"""
    example_5_tool_response = json.dumps([{'raw_institution_name': 'MIT',
                                           'authorships.institutions.id': 'institutions/I63966007', 
                                           'institutions.id': 'institutions/I63966007'}])
    example_5_answer = json.dumps({"filters": [
        {
            "id": "branch_work",
            "subjectEntity": "works",
            "type": "branch",
            "operator": "and",
            "children": [
                "leaf_1",
            ]
        },
        {
            "id": "leaf_1",
            "subjectEntity": "works",
            "type": "leaf",
            "column_id": "authorships.institutions.id",
            "operator": "is",
            "value": "institutions/I63966007"
        },
        {
            "id": "branch_institution",
            "subjectEntity": "institutions",
            "type": "branch",
            "column_id": "",
            "operator": "and",
            "value": "",
            "children": ['leaf_2']
        },
        {
            "id": "leaf_2",
            "subjectEntity": "works",
            "type": "leaf",
            "column_id": "authorships.institutions.continent",
            "operator": "is",
            "value": "continents/Q15"

        }
    ],
    "summarize_by": "institutions",
    "sort_by": {
    "column_id": "count",
    "direction": "desc"
    },
    "return_columns": [
        "display_name",
        "country_code",
        "ids.ror",
        "count",
        "mean(fwci)"
        ]
    })

    example_6 = "Which researchers collaborate with Stanford University the most?"
    example_6_tool = """ChatCompletionMessage(content=None, refusal=None, role='assistant', function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_JWHWkwhdOQhaVHqHYRAhCHA', function=Function(arguments='{"institution_name":"Stanford University"}', name='get_institution_id'), type='function')])"""
    example_6_tool_response = json.dumps([{'raw_institution_name': 'Stanford University',
                                           'authorships.institutions.id': 'institutions/I97018004', 
                                           'institutions.id': 'institutions/I97018004'}])
    example_6_answer = json.dumps({"filters": [
        {
            "id": "branch_work",
            "subjectEntity": "works",
            "type": "branch",
            "operator": "and",
            "children": [
                "leaf_1",
            ]
        },
        {
            "id": "leaf_1",
            "subjectEntity": "works",
            "type": "leaf",
            "column_id": "authorships.institutions.id",
            "operator": "is",
            "value": "institutions/I97018004"
        },
        {
            "id": "branch_author",
            "subjectEntity": "authors",
            "type": "branch",
            "column_id": "",
            "operator": "and",
            "value": "",
            "children": ['leaf_2']
        },
        {
            "id": "leaf_2",
            "subjectEntity": "works",
            "type": "leaf",
            "column_id": "authorships.institutions.id",
            "operator": "is not",
            "value": "institutions/I97018004"

        }
    ],
    "summarize_by": "authors",
    "sort_by": {
    "column_id": "count",
    "direction": "desc"
    },
    "return_columns": [
        "id",
        "ids.orcid",
        "display_name",
        "last_known_institutions.id",
        "count"
        ]
    })

    messages = [
        {"role": "system", 
         "content": "You are helping to take in database search requests from users for pulling data from OpenAlex and turn them into a JSON object. OpenAlex indexes scholarly works and their metadata."},
        {"role": "user", "content": information_for_system},
        {"role": "assistant", 
         "content": "I will refer back to this information when determining which columns need to be filtered, sorted, or returned"},
        {"role": "user","content": example_1}, 
        {"role": "user","content": example_1_answer}, 
        {"role": "user","content": example_2},    
        {"role": "assistant", "content": example_2_tool},
        {"role": "user", "content": example_2_tool_response},
        {"role": "assistant", "content": example_2_answer},
        {"role": "user", "content": example_3},
        {"role": "assistant", "content": example_3_answer},
        {"role": "user", "content": example_4}, 
        {"role": "assistant", "content": example_4_tool},
        {"role": "user", "content": example_4_tool_response},
        {"role": "assistant", "content": example_4_answer},
        {"role": "user", "content": example_5},
        {"role": "assistant", "content": example_5_tool},
        {"role": "user", "content": example_5_tool_response},
        {"role": "assistant", "content": example_5_answer},
        {"role": "user", "content": example_6},
        {"role": "assistant", "content": example_6_tool},
        {"role": "user", "content": example_6_tool_response},
        {"role": "assistant", "content": example_6_answer}]

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
        },
        {
            "type": "function",
            "function": {
                "name": "get_author_id",
                "description": "Get the OpenAlex author ID from the API when an author name needs to be looked up to find the ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "author_name": {
                            "type": "string",
                            "description": "The name of the author that needs to be looked up.",
                        },
                    },
                    "required": ["author_name"],
                    "additionalProperties": False,
                },
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_keyword_id",
                "description": "Get the OpenAlex keyword ID from the API when a specific subject or keyword needs to be looked up to find the ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_name": {
                            "type": "string",
                            "description": "A short phrase or word to look up in the OpenAlex keywords",
                        },
                    },
                    "required": ["search_name"],
                    "additionalProperties": False,
                },
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_source_id",
                "description": "Get the OpenAlex source ID from the API when a specific source needs to be looked up to find the ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_name": {
                            "type": "string",
                            "description": "The name of a journal, repository, or other to look up the OpenAlex source ID",
                        },
                    },
                    "required": ["search_name"],
                    "additionalProperties": False,
                },
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_funder_id",
                "description": "Get the OpenAlex funder ID from the API when a specific funding organization needs to be looked up to find the ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_name": {
                            "type": "string",
                            "description": "The name of a funding organization to look up the OpenAlex funder ID",
                        },
                    },
                    "required": ["search_name"],
                    "additionalProperties": False,
                },
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_publisher_id",
                "description": "Get the OpenAlex publisher ID from the API when a specific publishing organization needs to be looked up to find the ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_name": {
                            "type": "string",
                            "description": "The name of a publishing organization to look up the OpenAlex publisher ID",
                        },
                    },
                    "required": ["search_name"],
                    "additionalProperties": False,
                },
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_topic_id",
                "description": "Get the OpenAlex topic ID from the API when a specific topic needs to be looked up to find the ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_name": {
                            "type": "string",
                            "description": "The name of a topic look up the OpenAlex topic ID",
                        },
                    },
                    "required": ["search_name"],
                    "additionalProperties": False,
                },
            }
        }
    ]
    return tools

def get_all_entities_and_columns():
    entities_with_function_calling = ['institutions','authors','keywords','sources','funders','publishers','topics']
    entities_with_function_calling_not_set_up = ['concepts']
    entities_without_function_calling = ['works','continents', 'countries', 'domains','fields','institution-types','languages','licenses',
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


class ReturnColumnsObject(BaseModel):
    return_columns: list[str]

class SortByColumnsObject(BaseModel):
    sort_by: SortByObject

class ReturnSortByColumnsObject(BaseModel):
    sort_by: SortByObject
    return_columns: list[str]

class ParsedPromptObject(BaseModel):
    filters_needed: bool
    summarize_by: str
    summarize_by_filters_needed: bool
    sort_by_needed: bool
    return_columns_needed: bool

# # Define the two possible schemas
# class SchemaOne(Schema):
#     name = fields.Str(required=True)
#     age = fields.Int(required=True)

# class SchemaTwo(Schema):
#     title = fields.Str(required=True)
#     year = fields.Int(required=True)


# class FilterSchema(Schema):
#     items = fields.List(fields.Dict(), required=True)

#     @validates_schema
#     def validate_items(self, data, **kwargs):
#         errors = []
#         schema_one = LeafFiltersSchema()
#         schema_two = BranchFiltersSchema()

#         for i, item in enumerate(data['items']):
#             try:
#                 schema_one.load(item)
#             except ValidationError as err1:
#                 try:
#                     schema_two.load(item)
#                 except ValidationError as err2:
#                     errors.append({i: [err1.messages, err2.messages]})

#         if errors:
#             raise ValidationError({"items": errors})

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
#     filters = fields.List(FilterSchema())
#     summarize_by = fields.Str(nullable=True)
#     sort_by = fields.Nested(SortBySchema)
#     return_columns = fields.List(fields.Str())

#     class Meta:
#         ordered = True
