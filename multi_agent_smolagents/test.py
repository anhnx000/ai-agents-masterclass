from smolagents import HfApiModel, CodeAgent

agent = CodeAgent(model=HfApiModel(), tools=[], executor_type="docker")

agent.run("Can you give me the 100th Fibonacci number?")
