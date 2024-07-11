from e2b import Sandbox


with Sandbox(template="sever-worker") as sandbox:
    print(sandbox.commands.run("pwd").stdout)
    print(sandbox.commands.run("ls").stdout)
    sandbox.commands.run("cd /worker")
    print(sandbox.commands.run("ls").stdout)
