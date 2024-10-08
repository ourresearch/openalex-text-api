import functools
import json
import os
import random
import datetime
import requests
# import tiktoken
from typing import Union
from flask import jsonify
from openai import OpenAI
from pydantic import BaseModel, StrictStr, StrictBool, StrictFloat, StrictInt
from marshmallow import Schema, fields
from oqo_validate import OQOValidator

openai_model_version = "gpt-4o-2024-08-06"

# @functools.lru_cache(maxsize=64)
def get_openai_response(prompt):
    api_key = os.getenv("OPENAI_API_KEY")
    # api_key = config_vars['OPENAI_API_KEY']
    client = OpenAI(api_key=api_key)
    oql_entities = get_all_entities_and_columns()

    
    # Quick check for entity
    quick_entity = quick_entity_check(prompt, oql_entities)
    if quick_entity != "":
        return {"get_rows": quick_entity}


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

    # enc = tiktoken.encoding_for_model("gpt-4o")
    # print(len(enc.encode(json.dumps(messages_parsed))))

    completion = client.beta.chat.completions.parse(
            model=openai_model_version,
            messages=messages_parsed,
            response_format=ParsedPromptObject,
            temperature=0.2
        )
    
    parsed_prompt = json.loads(completion.choices[0].message.content)
    # print(parsed_prompt)

    # Getting examples to feed the model
    messages = example_messages_for_chat(oql_entities)

    if (not parsed_prompt['filter_works_needed'] and 
        not parsed_prompt['filter_aggs_needed'] and
        not parsed_prompt['sort_by_needed'] and 
        not parsed_prompt['show_columns_needed']):
        if parsed_prompt['get_rows'] == "":
            return {}
        else:
            json_object = {"get_rows": parsed_prompt['get_rows']}
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
    
    if (not parsed_prompt['filter_works_needed'] and 
        not parsed_prompt['filter_aggs_needed'] and
        not parsed_prompt['sort_by_needed'] and 
        parsed_prompt['show_columns_needed']):

        if parsed_prompt['get_rows'] == "":
            messages.append({"role": "user", "content": f"Give list of show columns for this query: {prompt}"})
        else:
            messages.append({"role": "user", "content": f"Give the list of show columns for the '{parsed_prompt['get_rows']}' entity in this query: {prompt}"})
            print(messages[-1])

        ok = False
        i = 0
        # retry the following code while expression is False
        while not ok:
            if i == 2:
                break
            completion = client.beta.chat.completions.parse(
                model=openai_model_version,
                messages=messages,
                response_format=ReturnColumnsObject,
                temperature=0.2
            )

            json_object = json.loads(completion.choices[0].message.content)
            if parsed_prompt['get_rows'] != "":
                json_object['get_rows'] = parsed_prompt['get_rows']
            ok, error_message = validator.validate(json_object)
            if 'show_columns' not in json_object:
                ok = False

            i+=1

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

    elif (not parsed_prompt['filter_works_needed'] and 
        not parsed_prompt['filter_aggs_needed'] and
        parsed_prompt['sort_by_needed'] and 
        not parsed_prompt['show_columns_needed']):

        if parsed_prompt['get_rows'] == "":
            messages.append({"role": "user", "content": f"Give the sort by columns for this query: {prompt}"})
        else:
            messages.append({"role": "user", "content": f"Give the sort by columns for the '{parsed_prompt['get_rows']}' entity in this query: {prompt}"})

        ok = False
        i = 0
        # retry the following code while expression is False
        while not ok:
            if i == 2:
                break
            completion = client.beta.chat.completions.parse(
                model=openai_model_version,
                messages=messages,
                response_format=SortByColumnsObject,
                temperature=0.2
            )

            json_object = json.loads(completion.choices[0].message.content)
            if parsed_prompt['get_rows'] != "":
                json_object['get_rows'] = parsed_prompt['get_rows']
            ok, error_message = validator.validate(json_object)
            if ('sort_by_column' not in json_object) or ('sort_by_order' not in json_object):
                ok = False

            i+=1

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
    
    elif (not parsed_prompt['filter_works_needed'] and 
        not parsed_prompt['filter_aggs_needed'] and
        parsed_prompt['sort_by_needed'] and 
        parsed_prompt['show_columns_needed']):

        if parsed_prompt['get_rows'] == "":
            messages.append({"role": "user", "content": f"Give the sort by and show columns for this query: {prompt}"})
        else:
            messages.append({"role": "user", "content": f"Give the sort by and show columns for the {parsed_prompt['get_rows']} entity in this query: {prompt}"})


        ok = False
        i = 0
        # retry the following code while expression is False
        while not ok:
            if i == 2:
                break
            completion = client.beta.chat.completions.parse(
                model=openai_model_version,
                messages=messages,
                response_format=ReturnSortByColumnsObject,
            )

            json_object = json.loads(completion.choices[0].message.content)
            if parsed_prompt['get_rows'] != "":
                json_object['get_rows'] = parsed_prompt['get_rows']
            ok, error_message = validator.validate(json_object)
            if not all(x in json_object.keys() for x in ['sort_by_column','sort_by_order', 
                                                         'show_columns']):
                ok = False

            i+=1

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
    if parsed_prompt['filter_aggs_final_output'] != "":
        messages.append({"role": "user", "content": f"{prompt} (make sure the filter_aggs include the filter for '{parsed_prompt['filter_aggs_final_output']}')"})
    else:
        messages.append({"role": "user", "content": prompt})


    # enc = tiktoken.encoding_for_model("gpt-4o")
    # print(len(enc.encode(json.dumps(messages))))

    ok = False
    i = 0
    # retry the following code 5 times while expression is False
    while not ok:
        if i == 3:
            break

        # Getting the tool needed for looking up new query
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            temperature=0.2
        )
        if response.choices[0].message.tool_calls:
            # Getting institution IDs (if needed)
            # print(response.choices[0].message.tool_calls)
            all_ids = use_openai_output_to_get_ids(response)

            # Giving data to model to get final json object
            messages.append({"role": "assistant", "content": str(response.choices[0].message)})
            messages.append({"role": "user", "content": json.dumps(all_ids)})

        completion = client.beta.chat.completions.parse(
            model=openai_model_version,
            messages=messages,
            response_format=OQLJsonObject,
            temperature=0.2
        )
        openai_json_object = json.loads(completion.choices[0].message.content)

        # print(openai_json_object)

        ok = True
        ok, error_message = validator.validate(openai_json_object)
        messages.append({"role": "assistant", "content": str(completion.choices[0].message.content)})
        messages.append({"role": "user", "content": f"That was not correct. The following error message was received:\n{error_message}\n\nPlease try again."})
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
        final_json_object = fix_output_for_final(openai_json_object, parsed_prompt)
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

def check_prompt_for_safety(prompt):
    if len(prompt) > 1000:
        return False
    else:
        return True

def quick_entity_check(prompt, oql_entities):
    
    singular_entity = [x[:-1] if x != 'countries' else 'country' for x in oql_entities.keys()]
    quick_check = \
        [x for x in oql_entities.keys()] + \
        [x[:-1] if x != 'countries' else 'country' for x in oql_entities.keys()] + \
        [f"get {x}" for x in oql_entities.keys()] + \
        [f"get {x}" for x in singular_entity]
    
    matching_entities = \
        [x for x in oql_entities.keys()]*4
    
    if " ".join(prompt.replace("-", " ").replace("!", "").replace(".", "")
                .replace("?", "").split(" ")).lower() in quick_check:
        # get index of matching_entities
        match = quick_check.index(" ".join(prompt.replace("-", " ").replace("!", "").replace(".", "")
                .replace("?", "").split(" ")).lower())
        entity = matching_entities[match]
        return entity
    else:
        return ""


def messages_for_parse_prompt(oql_entities):
    information_for_system = create_system_information(oql_entities)

    example_1 = "Show me all works in OpenAlex"
    example_1_answer = {
        "get_rows": "works",
        "filter_works_needed": False,
        "filter_works_final_output": "",
        "filter_aggs_needed": False,
        "filter_aggs_final_output": "",
        "sort_by_needed": False,
        "show_columns_needed": False
        }
    
    example_2 = "Show me all works from North Carolina State University in 2023 and show me the openalex ID, title, and cited by count. Show the highest cited publications first."
    example_2_answer = {
        "get_rows": "works",
        "filter_works_needed": True,
        "filter_works_final_output": "works from North Carolina State University in 2023",
        "filter_aggs_needed": False,
        "filter_aggs_final_output": "",
        "sort_by_needed": True,
        "show_columns_needed": True
        }
    
    example_3 = "Which institutions does NASA collaborate the most with in Africa?"
    example_3_answer = {
        "get_rows": "institutions",
        "filter_works_needed": True,
        "filter_works_final_output": "works by NASA",
        "filter_aggs_needed": True,
        "filter_aggs_final_output": "institutions in Africa",
        "sort_by_needed": True,
        "show_columns_needed": False
        }
    
    example_4 = "Which researchers at the University of Colorado have published the most work on SDG 13?"
    example_4_answer = {
        "get_rows": "authors",
        "filter_works_needed": True,
        "filter_works_final_output": "works on SDG 13",
        "filter_aggs_needed": True,
        "filter_aggs_final_output": "researchers at the University of Colorado",
        "sort_by_needed": True,
        "show_columns_needed": False
        }
    
    example_5 = "Which journals publish the highest cited research on coral bleaching?"
    example_5_answer = {
        "get_rows": "sources",
        "filter_works_needed": True,
        "filter_works_final_output": "works on coral bleaching",
        "filter_aggs_needed": True,
        "filter_aggs_final_output": "sources that are journals",
        "sort_by_needed": True,
        "show_columns_needed": False
        }
    
    example_6 = "authors or show me all authors or get authors"
    example_6_answer = {
        "get_rows": "authors",
        "filter_works_needed": False,
        "filter_works_final_output": "",
        "filter_aggs_needed": False,
        "filter_aggs_final_output": "",
        "sort_by_needed": False,
        "show_columns_needed": False
        }
    
    example_6 = "what are the work types in OpenAlex?"
    example_6_answer = {
        "get_rows": "types",
        "filter_works_needed": False,
        "filter_works_final_output": "",
        "filter_aggs_needed": False,
        "filter_aggs_final_output": "",
        "sort_by_needed": False,
        "show_columns_needed": False
        }
    
    example_7 = "what South African institutions are collaborating with MIT the most?"
    example_7_answer = {
        "get_rows": "institutions",
        "filter_works_needed": True,
        "filter_works_final_output": "works by MIT",
        "filter_aggs_needed": True,
        "filter_aggs_final_output": "South African institutions",
        "sort_by_needed": True,
        "show_columns_needed": False
        }
    
    example_8 = "which SDGs does Kyle Demes work on the most?"
    example_8_answer = {
        "get_rows": "sdgs",
        "filter_works_needed": True,
        "filter_works_final_output": "works by Kyle Demes",
        "filter_aggs_needed": False,
        "filter_aggs_final_output": "",
        "sort_by_needed": True,
        "show_columns_needed": False
        }
    
    example_9 = "which authors in Canada have the highest number of citations?"
    example_9_answer = {
        "get_rows": "authors",
        "filter_works_needed": False,
        "filter_works_final_output": "",
        "filter_aggs_needed": True,
        "filter_aggs_final_output": "authors in Canada",
        "sort_by_needed": True,
        "show_columns_needed": False
        }
    
    example_10 = "which institutions in Japan have the highest number of works?"
    example_10_answer = {
        "get_rows": "institutions",
        "filter_works_needed": False,
        "filter_works_final_output": "",
        "filter_aggs_needed": True,
        "filter_aggs_final_output": "institutions in Japan",
        "sort_by_needed": True,
        "show_columns_needed": False
        }
    
    example_11 = "which researchers collaborate with Stanford University?"
    example_11_answer = {
        "get_rows": "authors",
        "filter_works_needed": True,
        "filter_works_final_output": "works by Stanford University",
        "filter_aggs_needed": True,
        "filter_aggs_final_output": "researchers not working at Stanford University",
        "sort_by_needed": False,
        "show_columns_needed": False
        }
    
    example_12 = "which researchers at Boston University are collaborating with researchers in Spain?"
    example_12_answer = {
        "get_rows": "authors",
        "filter_works_needed": True,
        "filter_works_final_output": "works by Spain",
        "filter_aggs_needed": True,
        "filter_aggs_final_output": "researchers at Boston University",
        "sort_by_needed": False,
        "show_columns_needed": False
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
        {"role": "user","content": json.dumps(example_6_answer)},
        {"role": "user","content": example_7}, 
        {"role": "user","content": json.dumps(example_7_answer)},
        {"role": "user","content": example_8}, 
        {"role": "user","content": json.dumps(example_8_answer)},
        {"role": "user","content": example_9}, 
        {"role": "user","content": json.dumps(example_9_answer)},
        {"role": "user","content": example_10}, 
        {"role": "user","content": json.dumps(example_10_answer)},
        {"role": "user","content": example_11}, 
        {"role": "user","content": json.dumps(example_11_answer)},
        {"role": "user","content": example_12}, 
        {"role": "user","content": json.dumps(example_12_answer)}
    ]
    return messages

def fix_output_for_final(old_json_object, parsed_prompt):

    json_object = old_json_object.copy()

    # check all work filters for 'is' operator and if present, remove the operator key
    if json_object.get('filter_works'):
        for filter_obj in json_object['filter_works']:
            if filter_obj['operator'] == "is":
                filter_obj.pop('operator')
            elif filter_obj['operator'] in ['>', '<', '>=','<=']:
                filter_obj['operator'] = f"is {filter_obj['operator'].replace('>=', 'greater than or equal to').replace('<=', 'less than or equal to').replace('>', 'greater than').replace('<', 'less than')}"

            if filter_obj.get('operator') and filter_obj['operator'] == "is greater than or equal to":
                filter_obj['value'] = filter_obj['value'] - 1
                filter_obj['operator'] = "is greater than"
            elif filter_obj.get('operator') and filter_obj['operator'] == "is less than or equal to":
                filter_obj['value'] = filter_obj['value'] + 1
                filter_obj['operator'] = "is less than"
    else:
        _ = json_object.pop('filter_works')
    
    # check all agg filters for 'is' operator and if present, remove the operator key
    if json_object.get('filter_aggs'):
        if not parsed_prompt['filter_aggs_needed']:
            _ = json_object.pop('filter_aggs')
        else:
            for filter_obj in json_object['filter_aggs']:
                if filter_obj['operator'] == "is":
                    filter_obj.pop('operator')
                elif filter_obj['operator'] in ['>', '<', '>=','<=']:
                    filter_obj['operator'] = f"is {filter_obj['operator'].replace('>=', 'greater than or equal to').replace('<=', 'less than or equal to').replace('>', 'greater than').replace('<', 'less than')}"

            if filter_obj.get('operator') and filter_obj['operator'] == "is greater than or equal to":
                filter_obj['value'] = filter_obj['value'] - 1
                filter_obj['operator'] = "is greater than"
            elif filter_obj.get('operator') and filter_obj['operator'] == "is less than or equal to":
                filter_obj['value'] = filter_obj['value'] + 1
                filter_obj['operator'] = "is less than"
    else:
         _ = json_object.pop('filter_aggs')

    if not parsed_prompt['sort_by_needed']:
        _ = json_object.pop('sort_by_column')
        _ = json_object.pop('sort_by_order')

    if not parsed_prompt['show_columns_needed']:
        _ = json_object.pop('show_columns')

    # if not json_object.get('filter_works'):
        # if not json_object['filter_works']:
        
    
    # if not json_object.get('filter_aggs'):
        # if not json_object['filter_aggs']:
    
    return json_object 

# def fix_output_for_final(old_json_object):
#     json_object = process_both_filters(old_json_object)

#     for filter_obj in json_object['filters']:
#         if filter_obj['value'] == "":
#             filter_obj['value'] = None
            
#         if filter_obj['column_id'] == "":
#             filter_obj['column_id'] = None
    
#     if json_object['summarize_by'] == "":
#         json_object['summarize_by'] = None
#     elif json_object['summarize_by'] == "works":
#         json_object['summarize_by'] = "all"

#     final_filter_obj = []
#     branch_objs = 0
#     for filter_obj in json_object['filters']:
#         if filter_obj['type'] == "branch":
#             if branch_objs == 0:
#                 if (filter_obj['subjectEntity'] != "works") and not json_object['summarize_by']:
#                     filter_obj['subjectEntity'] = "works"
#             branch_objs += 1
#             final_filter_obj.append({k: v for k, v in filter_obj.items() if k not in ['value','column_id']})
#         elif filter_obj['type'] == "leaf":
#             if filter_obj['operator'] in ['>', '<', '>=','<=']:
#                 filter_obj['operator'] = f"is {filter_obj['operator'].replace('>=', 'greater than or equal to').replace('<=', 'less than or equal to').replace('>', 'greater than').replace('<', 'less than')}"
#             if isinstance(filter_obj['value'], str):
#                 if 'works' in filter_obj['value']:
#                     if 'works/W' in filter_obj['value']:
#                         filter_obj['value'] = filter_obj['value'].split("works/W")[1]
#             final_filter_obj.append({k: v for k, v in filter_obj.items() if k not in ['children']})

#     # check if final_filter_obj is leaf only
#     if (branch_objs == 0) and (len(final_filter_obj) >= 1):
#         new_final_filter_obj = [
#             {
#                 "id": "branch_work",
#                 "subjectEntity": "works",
#                 "type": "branch",
#                 "operator": "and",
#                 "children": [x['id'] for x in final_filter_obj if x['subjectEntity'] == "works"]
#             }
#         ]
#         _ = [new_final_filter_obj.append(x) for x in final_filter_obj]
#         final_filter_obj = new_final_filter_obj

#     json_object['filters'] = final_filter_obj
#     return json_object 

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
                                    "authorships.author.id": f"authors/{author_id}", 
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
            
        # elif tool_call.function.name == "get_publisher_id":
        #     search_name = arguments.get('search_name')
        
        #     publisher_id = get_publisher_id(search_name)

        #     all_tool_data.append({"raw_search_name": search_name, 
        #                           "primary_location.source.publisher_lineage": f"publishers/{publisher_id}"})
            
        # elif tool_call.function.name == "get_topic_id":
        #     search_name = arguments.get('search_name')
        
        #     topic_id = get_topic_id(search_name)

        #     all_tool_data.append({"raw_search_name": search_name, 
        #                           "primary_topic.id": f"topics/{topic_id}"})
    return all_tool_data

def create_system_information(entities_info):
    # get current year
    system_info = f"The year is {str(datetime.datetime.now().year)}. Please keep that in mind when responding.\n\n"
    system_info += "If the name of an institution is given, use the get_institution_id tool in order to retrieve the OpenAlex institution ID.\n\n"
    system_info += "If the name of an author is given, use the get_author_id tool in order to retrieve the OpenAlex author ID.\n\n"
    system_info += "If the name of a keyword is given, use the get_keyword_id tool in order to retrieve the OpenAlex keyword ID.\n\n"
    system_info += "If the name of a source is given, use the get_source_id tool in order to retrieve the OpenAlex source ID.\n\n"
    system_info += "If the name of a funder is given, use the get_funder_id tool in order to retrieve the OpenAlex funder ID.\n\n"
    # system_info += "If the name of a publisher is given, use the get_publisher_id tool in order to retrieve the OpenAlex publisher ID.\n\n"
    # system_info += "If the name of a topic is given, use the get_topic_id tool in order to retrieve the OpenAlex topic ID.\n\n"
    system_info += "The value for country or country_code is the 2 letter representation of that country.\n\n"
    system_info += "Filter operator must be one of the following: ['is','is not','is greater than','is less than']\n\n"
    system_info += "Default no sort_by_column unless another sort_by_column is specified by the user.\n\n"
    system_info += "Please look at the following subjectEntity information to see which columns can be sorted or filtered or returned and also which ones need to use a function call tool in order to look up the entity:\n\n"
    for entity in entities_info.keys():
        if entity == "works":
            system_info += f"subjectEntity: {entity}\n\n"
            system_info += f"Columns (column_id) in {entity} that can be filtered (filter_works):\n"
            for col in entities_info[entity]['filter_works']:
                system_info += f"{col}: {entities_info[entity]['columns'][col]}\n"
            system_info += f"\n"
            system_info += f"Columns (column_id) in {entity} that can be sorted (sort_by_column):\n"
            for col in entities_info[entity]['sort_by_columns']:
                system_info += f"{col}: {entities_info[entity]['columns'][col]}\n"
            system_info += f"\n"
            system_info += f"Columns (column_id) in {entity} that can be shown (show_columns):\n"
            for col in entities_info[entity]['show_columns']:
                system_info += f"{col}: {entities_info[entity]['columns'][col]}\n"
            system_info += f"\n"
            system_info += f"Function call tool needed for {entity}: Yes\n\n\n\n"
        elif entities_info[entity]['function_call']:
            system_info += f"subjectEntity: {entity}\n\n"
            system_info += f"Columns (column_id) in {entity} that can be filtered (filter_aggs):\n"
            for col in entities_info[entity]['filter_aggs']:
                system_info += f"{col}: {entities_info[entity]['columns'][col]}\n"
            system_info += f"\n"
            system_info += f"Columns (column_id) in {entity} that can be sorted (sort_by_column):\n"
            for col in entities_info[entity]['sort_by_columns']:
                system_info += f"{col}: {entities_info[entity]['columns'][col]}\n"
            system_info += f"\n"
            system_info += f"Columns (column_id) in {entity} that can be shown (show_columns):\n"
            for col in entities_info[entity]['show_columns']:
                system_info += f"{col}: {entities_info[entity]['columns'][col]}\n"
            system_info += f"\n"
            system_info += f"Function call tool needed for {entity}: Yes\n\n\n\n"
        else:
            system_info += f"subjectEntity: {entity}\n\n"
            system_info += f"Columns (column_id) in {entity} that can be filtered (filter_aggs):\n"
            for col in entities_info[entity]['filter_aggs']:
                system_info += f"{col}: {entities_info[entity]['columns'][col]}\n"
            system_info += f"\n"
            system_info += f"Columns (column_id) in {entity} that can be sorted (sort_by_column):\n"
            for col in entities_info[entity]['sort_by_columns']:
                system_info += f"{col}: {entities_info[entity]['columns'][col]}\n"
            system_info += f"\n"
            system_info += f"Columns (column_id) in {entity} that can be shown (show_columns):\n"
            for col in entities_info[entity]['show_columns']:
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
        "get_works": "works",
        "filter_works": [],
        "filter_aggs": [],
        "sort_by_column": "",
        "sort_by_order": "",
        "show_columns": []})
    
    example_1a = "Just list all of the work types in OpenAlex (also known as 'get types')"
    example_1a_answer = json.dumps({
        "get_rows": "types",
        "filter_works": [],
        "filter_aggs": [],
        "sort_by_column": "",
        "sort_by_order": "",
        "show_columns": []})

    example_1b = "What respositories are indexed in OpenAlex?"
    example_1b_answer = json.dumps({
        "get_rows": "sources",
        "filter_works": [],
        "filter_aggs": [
            {
                "column_id": "source_type",
                "operator": "is",
                "value": "source-types/repositories"
            }
        ],
        "sort_by_column": "",
        "sort_by_order": "",
        "show_columns": []})
    
    example_1c = "Give me a one-line summary ('summarize all') of all works in OpenAlex"
    example_1c_answer = json.dumps({
        "get_rows": "summary",
        "filter_works": [],
        "filter_aggs": [],
        "sort_by_column": "",
        "sort_by_order": "",
        "show_columns": []})

    example_2 = "List all works from North Carolina State University (using the OpenAlex ID) since 2023 and show me the openalex ID, title, and cited by count. Show the highest cited publications first."
    example_2_tool = """ChatCompletionMessage(content=None, refusal=None, role='assistant', function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_fcEKw4AeBTklT7HtJyakgboc', function=Function(arguments='{"institution_name":"North Carolina State University"}', name='get_institution_id'), type='function')])"""
    example_2_tool_response = json.dumps([{'raw_institution_name': 'North Carolina State University',
                                           'authorships.institutions.id': 'institutions/I137902535',
                                           'institutions.id': 'institutions/I137902535'}])
    example_2_answer = json.dumps({
        "get_rows": "works",
        "filter_works":
        [{
            "column_id": "publication_year",
            "operator": "is greater than or equal to",
            "value": 2023
        },
        {
            "column_id": "authorships.institutions.id",
            "operator": "is",
            "value": "institutions/I137902535"
        }],
        "filter_aggs": [],
        "sort_by_column": "cited_by_count",
        "sort_by_order": "desc",
        "show_columns": [
            "openalex_id",
            "paper_title",
            "cited_by_count"
        ]})


    example_3 = "Give me high level information for French institutions (aggregate) (make sure the filter_aggs include the filter for 'French institutions')"
    example_3_answer = json.dumps({
        "get_rows": "institutions",
        "filter_works": [],
        "filter_aggs": [
           {
                "column_id": "country_code",
                "operator": "is",
                "value": "countries/FR"
            }
        ],
        "sort_by_column": "",
        "sort_by_order": "",
        "show_columns": []
        })

    example_4 = "I want to see all works from Sorbonne University that are open access and in English while also being tagged with the SDG for good health and well-being."
    example_4_tool = """ChatCompletionMessage(content=None, refusal=None, role='assistant', function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_DOSHfhsdiSFhsFHsAH', function=Function(arguments='{"institution_name":"Sorbonne University"}', name='get_institution_id'), type='function')])"""
    example_4_tool_response = json.dumps([{'raw_institution_name': 'Sorbonne University',
                                           'authorships.institutions.id': 'institutions/I39804081',
                                           'institutions.id': 'institutions/I39804081'}])
    example_4_answer = json.dumps({
    "get_rows": "works",
    "filter_works": [
        {
            "column_id": "authorships.institutions.id",
            "operator": "is",
            "value": "institutions/I39804081"
        },
        {
            "column_id": "open_access.is_oa",
            "operator": "is",
            "value": True
        },
        {
            "column_id": "language",
            "operator": "is",
            "value": "languages/en"
        },
        {
            "column_id": "sustainable_development_goals.id",
            "operator": "is",
            "value": "sdgs/3"
        }
    ],
    "filter_aggs": [],
    "sort_by_column": "",
    "sort_by_order": "",
    "show_columns": []
    })

    example_5 = "Show me South African institutions that collaborate with MIT the most. (make sure the filter_aggs include the filter for 'South African institutions')"
    example_5_tool = """ChatCompletionMessage(content=None, refusal=None, role='assistant', function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_DOSHfhsdiSFhsFHsAH', function=Function(arguments='{"institution_name":"MIT"}', name='get_institution_id'), type='function')])"""
    example_5_tool_response = json.dumps([{'raw_institution_name': 'MIT',
                                           'authorships.institutions.id': 'institutions/I63966007',
                                           'institutions.id': 'institutions/I63966007'}])
    example_5_answer = json.dumps(
    {
        "get_rows": "institutions",
        "filter_works": [
        {
                "column_id": "authorships.countries",
                "operator": "is",
                "value": "countries/ZA"
            },
            {
                "column_id": "authorships.institutions.id",
                "operator": "is",
                "value": "institutions/I63966007"
            }
    ],
    "filter_aggs": [
        {
            "column_id": "country_code",
            "operator": "is",
            "value": "countries/ZA"

        }
    ],
    "sort_by_column": "count",
    "sort_by_order": "desc",
    "show_columns": []
    })

    example_6 = "Which researchers collaborate with Stanford University the most? "
    example_6_tool = """ChatCompletionMessage(content=None, refusal=None, role='assistant', function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_JWHWkwhdOQhaVHqHYRAhCHA', function=Function(arguments='{"institution_name":"Stanford University"}', name='get_institution_id'), type='function')])"""
    example_6_tool_response = json.dumps([{'raw_institution_name': 'Stanford University',
                                           'authorships.institutions.id': 'institutions/I97018004',
                                           'institutions.id': 'institutions/I97018004'}])
    example_6_answer = json.dumps(
        {
            "get_rows": "authors",
            "filter_works": [
                {
                    "column_id": "authorships.institutions.id",
                    "operator": "is",
                    "value": "institutions/I97018004"
                }
            ],
            "filter_aggs": [
                {
                    "column_id": "affiliations.institution.id",
                    "operator": "is not",
                    "value": "institutions/I97018004"

                }
            ],
            "sort_by_column": "count",
            "sort_by_order": "desc",
            "show_columns": [
                "id",
                "ids.orcid",
                "display_name",
                "last_known_institutions.id",
                "count"
            ]
        })
    
    example_7 = "Which researchers at virginia tech are currently collaborating with researchers in Ukraine?"
    example_7_tool = """ChatCompletionMessage(content=None, refusal=None, role='assistant', function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_JWHWkwhdOQhaVHqHYRAhCHA', function=Function(arguments='{"institution_name":"Virginia Tech"}', name='get_institution_id'), type='function')])"""
    example_7_tool_response = json.dumps([{'raw_institution_name': 'Virginia Tech',
                                           'authorships.institutions.id': 'institutions/I859038795',
                                           'institutions.id': 'institutions/I859038795'}])
    example_7_answer = json.dumps(
        {
            "get_rows": "authors",
            "filter_works": [
                {
                    "column_id": "authorships.countries",
                    "operator": "is",
                    "value": "countries/UA"
                },
                {
                    "column_id": "authorships.institutions.id",
                    "operator": "is",
                    "value": "institutions/I859038795"
                }
            ],
            "filter_aggs": [
                {
                    "column_id": "last_known_institutions.id",
                    "operator": "is",
                    "value": "institutions/I859038795"
                }
            ],
            "sort_by_column": "count",
            "sort_by_order": "desc",
            "show_columns": [
                "id",
                "ids.orcid",
                "display_name",
                "last_known_institutions.id",
                "count"
            ]
        })
    
    example_8 = "which SDGs does Kyle Demes work on the most?"
    example_8_tool = """ChatCompletionMessage(content=None, refusal=None, role='assistant', function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_JWHWkwhdOQhaVHqHYRAhCHA', function=Function(arguments='{"author_name":"Kyle Demes"}', name='get_author_id'), type='function')])"""
    example_8_tool_response = json.dumps([{'raw_author_name': 'Kyle Demes',
                                           'authorships.author.id': 'authors/A5086928770',
                                           'authors.id': 'authors/A5086928770'}])
    
    example_8_answer = json.dumps(
        {
            "get_rows": "sdgs",
            "filter_works": [
                {
                    "column_id": "authorships.author.id",
                    "operator": "is",
                    "value": "authors/A5086928770"
                }
            ],
            "filter_aggs": [],
            "sort_by_column": "count",
            "sort_by_order": "desc",
            "show_columns": []
        })

    messages = [
        {"role": "system",
         "content": "You are helping to take in database search requests from users for pulling data from OpenAlex and turn them into a JSON object. OpenAlex indexes scholarly works and their metadata."},
        {"role": "user", "content": information_for_system},
        {"role": "assistant",
         "content": "I will refer back to this information when determining which columns need to be filtered, sorted, or returned"},
         {"role": "user", "content": "If the name of an institution, author, keyword, source, funder, publisher, or topic is given, use the appropriate tool in order to retrieve the OpenAlex ID."},
        {"role": "assistant",
         "content": "I will make use of the tools to look up the IDs for the entities in the OpenAlex database."},
        {"role": "user","content": example_1},
        {"role": "user","content": example_1_answer},
        {"role": "user","content": example_1a},
        {"role": "user","content": example_1a_answer},
        {"role": "user","content": example_1b},
        {"role": "user","content": example_1b_answer},
        {"role": "user","content": example_1c},
        {"role": "user","content": example_1c_answer},
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
        {"role": "assistant", "content": example_6_answer},
        {"role": "user", "content": example_7},
        {"role": "assistant", "content": example_7_tool},
        {"role": "user", "content": example_7_tool_response},
        {"role": "assistant", "content": example_7_answer},
        {"role": "user", "content": example_8},
        {"role": "assistant", "content": example_8_tool},
        {"role": "user", "content": example_8_tool_response},
        {"role": "assistant", "content": example_8_answer}]

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
    entities_without_function_calling = ['continents', 'countries', 'domains','fields','institution-types','languages','licenses',
                                         'sdgs','source-types','subfields','types']
    
    config_json = requests.get("https://api.openalex.org/entities/config").json()

    oql_info = {}
    for key in config_json.keys():
        if key == 'works':
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
            oql_info[key]['filter_works'] = cols_for_filter
            oql_info[key]['sort_by_columns'] = cols_for_sort_by
            oql_info[key]['show_columns'] = cols_for_return
        elif key in entities_with_function_calling:
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
            oql_info[key]['filter_aggs'] = cols_for_filter
            oql_info[key]['sort_by_columns'] = cols_for_sort_by
            oql_info[key]['show_columns'] = cols_for_return
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
                
            oql_info[key]['filter_aggs'] = cols_for_filter
            oql_info[key]['sort_by_columns'] = cols_for_sort_by
            oql_info[key]['show_columns'] = cols_for_return
            

    return oql_info

class FilterObject(BaseModel):
    column_id: str
    operator: str
    value: Union[StrictStr, StrictBool, StrictFloat, StrictInt] = None

class OQLJsonObject(BaseModel):
    get_rows: str
    filter_works: list[FilterObject]
    filter_aggs: list[FilterObject]
    sort_by_column: str
    sort_by_order: str
    show_columns: list[str]
    
class ReturnColumnsObject(BaseModel):
    show_columns: list[str]

class SortByColumnsObject(BaseModel):
    sort_by_column: str
    sort_by_order: str

class ReturnSortByColumnsObject(BaseModel):
    sort_by_column: str
    sort_by_order: str
    show_columns: list[str]

class ParsedPromptObject(BaseModel):
    get_rows: str
    filter_works_needed: bool
    filter_works_final_output: str
    filter_aggs_needed: bool
    filter_aggs_final_output: str
    sort_by_needed: bool
    show_columns_needed: bool
