import argparse

# 1. 创建解析器实例
# description 参数会在帮助信息中显示，用于解释脚本的整体用途
parser = argparse.ArgumentParser(description="这是一个简单的文件处理脚本。")

# 2. 添加参数定义

# --- 位置参数示例 ---
# 位置参数是必须提供的，且顺序很重要
# 'input_file' 是参数名，用户在命令行中直接输入文件名
parser.add_argument("input_file", type=str,
                    help="要处理的输入文件路径。")

# --- 可选参数示例 ---
# 可选参数以 '-' 或 '--' 开头
# action='store_true' 意味着如果提供了这个参数，则它的值为 True，否则为 False
parser.add_argument("--verbose", "-v", action="store_true",
                    help="启用详细输出模式。")

# type 参数指定了参数的预期类型，argparse 会自动进行类型转换
# default 参数为可选参数设置默认值
parser.add_argument("--output", "-o", type=str, default="output.txt",
                    help="输出文件路径。默认为 output.txt")

# choices 参数限制了参数的合法取值范围
parser.add_argument("--mode", type=str, choices=["read", "write", "append"], default="read",
                    help="操作模式：'read', 'write' 或 'append'。默认为 'read'。")

# nargs 参数用于指定参数接受的值的数量
# nargs='+' 表示至少一个值
parser.add_argument("--tags", nargs='+', type=str,
                    help="要添加的标签列表 (至少一个)。")

# 3. 解析命令行参数
# parse_args() 会从 sys.argv 中读取命令行参数，并根据您定义的规则进行解析
args = parser.parse_args()

# 4. 使用解析后的参数
print(f"--- 解析结果 ---")
print(f"输入文件: {args.input_file}")
print(f"详细模式: {args.verbose}")
print(f"输出文件: {args.output}")
print(f"操作模式: {args.mode}")
print(f"标签: {args.tags}")

print("\n--- 脚本逻辑模拟 ---")
if args.verbose:
    print("详细模式已启用。")
print(f"正在以 '{args.mode}' 模式处理文件: {args.input_file}")
if args.tags:
    print(f"处理中包含以下标签: {', '.join(args.tags)}")
print(f"结果将写入: {args.output}")

'''
(CS336) (base) mutyuu@steve:~/workspace/LLM_A2Z/examples_hf$ python example_argparser.py config.json --mode append --tags configuration json important
--- 解析结果 ---
输入文件: config.json
详细模式: False
输出文件: output.txt
操作模式: append
标签: ['configuration', 'json', 'important']

--- 脚本逻辑模拟 ---
正在以 'append' 模式处理文件: config.json
处理中包含以下标签: configuration, json, important
结果将写入: output.txt
(CS336) (base) mutyuu@steve:~/workspace/LLM_A2Z/examples_hf$ python example_argparser.py config.json --mode append --tags 1 2 3
--- 解析结果 ---
输入文件: config.json
详细模式: False
输出文件: output.txt
操作模式: append
标签: ['1', '2', '3']

--- 脚本逻辑模拟 ---
正在以 'append' 模式处理文件: config.json
处理中包含以下标签: 1, 2, 3
结果将写入: output.txt
'''