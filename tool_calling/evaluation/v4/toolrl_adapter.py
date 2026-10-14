import json

from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from overrides import override


json_string = """{\"name\": \"Tool name\", \"parameters\": {\"Parameter name\": \"Parameter content\", \"... ...\": \"... ...\"}}
{\"name\": \"... ...\", \"parameters\": {\"... ...\": \"... ...\", \"... ...\": \"... ...\"}}"""

# Define the system prompt as a constant
SYS = """You are a helpful multi-turn dialogue assistant capable of leveraging tool calls to solve user tasks and provide structured chat responses.

**Available Tools**
In your response, you can use the following tools:
{tools}

**Steps for Each Turn**
1. **Think:** Recall relevant context and analyze the current user goal.
2. **Decide on Tool Usage:** If a tool is needed, specify the tool and its parameters.
3. **Respond Appropriately:** If a response is needed, generate one while maintaining consistency across user queries.

**Output Format**
```plaintext
<think> Your thoughts and reasoning </think>
<tool_call>
{json_string}
...
</tool_call>
<response> AI's final response </response>
```

**Important Notes**
1. You must always include the `<think>` field to outline your reasoning. Provide at least one of `<tool_call>` or `<response>`. Decide whether to use `<tool_call>` (possibly multiple times), `<response>`, or both.
2. You can invoke multiple tool calls simultaneously in the `<tool_call>` fields. Each tool call should be a JSON object with a "name" field and an "parameters" field containing a dictionary of parameters. If no parameters are needed, leave the "parameters" field an empty dictionary.
3. Refer to the previous dialogue records in the history, including the user's queries, previous `<tool_call>`, `<response>`, and any tool feedback noted as `<obs>` (if exists).
"""


class RLLAHandler(OSSHandler):
    def __init__(self, model_name, temperature, registry_name='rlla-eval', is_fc_model=False, **kwargs) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)

    @override
    def _format_prompt(self, messages, function, turn_type="single_turn"):
        """
        "chat_template": "{%- set system_message = 'You are a helpful assistant.' %}\n{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content'] %}\n    {%- if messages[1]['role'] == 'system' %}\n        {%- set format_message = messages[1]['content'] %}\n        {%- set loop_messages = messages[2:] %}\n    {%- else %}\n        {%- set loop_messages = messages[1:] %}\n    {%- endif %}\n{%- else %}\n    {%- set loop_messages = messages %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n{%- if system_message is defined %}\n{{- '<|im_start|>system\n' + system_message + '<|im_end|>\n' }}\n{%- endif %}\n\n\n{%- if tools is not none %}\n{% set task_instruction %}You are a tool calling assistant. In order to complete the user's request, you need to select one or more appropriate tools from the following tools and fill in the correct values for the tool parameters. Your specific tasks are:\n1. Make one or more function/tool calls to meet the request based on the question.\n2. If none of the function can be used, point it out and refuse to answer.\n3. If the given question lacks the parameters required by the function, also point it out.\n\nThe following are characters that may interact with you\n1. user: Provides query or additional information.\n2. tool: Returns the results of the tool calling.\n{% endset %}\n\n{% set format_instruction %}\nThe output MUST strictly adhere to the following JSON format, and NO other text MUST be included.\nThe example format is as follows. Please make sure the parameter type is correct. If no function call is needed, please directly output an empty list '[]'\n```\n[\n    {\"name\": \"func_name1\", \"arguments\": {\"argument1\": \"value1\", \"argument2\": \"value2\"}},\n    ... (more tool calls as required)\n]\n```\n{% endset %}\n{{- '<|im_start|>user\n[BEGIN OF TASK INSTRUCTION]\n' + task_instruction + '\n[END OF TASK INSTRUCTION]\n\n'}}\n    {{- '[BEGIN OF AVAILABLE_TOOLS]\n' }}\n    {{- tools|string }}\n    {{- '\n[END OF AVAILABLE_TOOLS]\n\n' }}\n    {{- '\n[BEGIN OF TASK INSTRUCTION]\n' + format_instruction + '\n[END OF TASK INSTRUCTION]\n\n<|im_end|>\n' }}\n{%- endif %}\n\n{%- for message in loop_messages %}\n    {%- set role = message['role'] %}\n    {%- set content = message['content'] %}\n    {{- '<|im_start|>'+ role +'\n' +  content + '<|im_end|>\n'}}\n{%- endfor %}\n{{- '<|im_start|>assistant\n' }}",
        """

        def convert_to_format_tool(tools, count=1):
            if isinstance(tools, dict):
                format_tools = {
                    "name": tools["name"],
                    "description": tools["description"],
                    "parameters": tools["parameters"].get("properties", {}),
                }
                tool_string = f"{count}. Name: {format_tools['name']}\nDescription: {format_tools['description']}\nParameters: {json.dumps(format_tools['parameters'])}"
                return tool_string
            elif isinstance(tools, list):
                tool_list = [convert_to_format_tool(tool, idx+1) for idx, tool in enumerate(tools)]
                return "\n".join(tool_list)
            else:
                return tools

        tools = convert_to_format_tool(function)

        SYSTEM_PROMPT = SYS.format(tools=tools, json_string=json_string)
        
        USER_PROMPT = "**Dialogue Records History**\n"
        
        for msg_idx, message in enumerate(messages):
            if message["role"] == "system":
                continue
            
            elif message["role"] == "user":
                if turn_type == "multi_turn":
                    USER_PROMPT += f"<user> {message['content'].strip()}\nUse the one or more necessary tool calls to complete the task. You could perform tool calls for multiple rounds so you can try and error. Please make a comprehensive plan about how to achieve the goal step by step, and begin to call the tool step by step. If no tools apply or required parameters are missing, please also directly state this in your response without tool calls. </user>\n"
                else:
                    USER_PROMPT += f"<user> {message['content'].strip()}\nIf there's no appropriate tools to apply or required parameters are missing, please directly inform me in your response without any tool call, or call the tool with the name as 'None'. Otherwise, you should use one or more necessary tool calls to complete the given task in this turn. </user>\n"
                
            elif message["role"] == "tool":
                tool_name = message["name"].strip()
                tool_result = message["content"].strip()
                USER_PROMPT += f"<obs> You have made the tool call {tool_name}. Execution returns: {tool_result} </obs>\n"
                if msg_idx == len(messages) - 1:
                    USER_PROMPT += f"\n<user> If you think you have completed the current task, or the task cannot be finished, please respond directly without additional tool calls. If you encounter an error during tool execution or the task remains unfinished, retry with the one or more necessary tool calls according to your thought and plan until completion. Based on the tool execution feedback, reflect on if understanding or selectioin of tool is wrong, what tool calling step is missing, and how to achieve the task goal from now on. </user>\n"
                    
            elif message["role"] == "assistant":
                # we make the thought shorter and more concise in the history
                content = message["content"].strip()
                message["content"] = content
                USER_PROMPT += f"\n{message['content'].strip()}\n"
        
        USER_PROMPT = USER_PROMPT.strip()
        
        return f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{USER_PROMPT}<|im_end|>\n<|im_start|>assistant\n"

    @override
    def decode_ast(self, result, language="Python", has_tool_call_tag=False):
        if "<tool_call>" not in result:
            return []
        
        tool_calls = result.split("<tool_call>")[-1].split("</tool_call>")[0].strip().split("\n")
        
        decoded_output = []
        for tool_call in tool_calls:
            try:
                tool_call = json.loads(tool_call)
                tool_call = self.parse_parameters(tool_call)
                name = tool_call["name"]
                if name.strip().lower() == "none":
                    continue
                arguments = tool_call["parameters"]
                decoded_output.append({name: arguments})
            except:
                continue

        return decoded_output

    @override
    def decode_execute(self, result, has_tool_call_tag=False):
        if "<tool_call>" not in result:
            return []
        
        calls = result.split("<tool_call>")[-1].split("</tool_call>")[0].strip().split("\n")
        tool_calls = []
        for call in calls:
            try:
                tool_call = json.loads(call)
                tool_call = self.parse_parameters(tool_call)
                tool_calls.append(tool_call)
            except:
                continue

        function_call = self.xlam_json_to_python_tool_calls(tool_calls)
        return function_call
    
    @staticmethod
    def xlam_json_to_python_tool_calls(tool_calls):
        """
        Converts a list of function calls in xLAM JSON format to Python format.

        Parameters:
        tool_calls (list): A list of dictionaries, where each dictionary represents a function call in JSON format.

        Returns:
        python_format (list): A list of strings, where each string is a function call in Python format.
        """
        if not isinstance(tool_calls, list):
            tool_calls = [tool_calls]

        python_format = []
        for tool_call in tool_calls:
            if isinstance(tool_call, dict):
                name = tool_call.get("name", "")
                arguments = tool_call.get("parameters", {})
                if name.strip().lower() == "none":
                    continue
                args_str = ", ".join(
                    [f"{key}={repr(value)}" for key, value in arguments.items()]
                )
                python_format.append(f"{name}({args_str})")

        return python_format

    @staticmethod
    def parse_parameters(tool_call):
        return tool_call
