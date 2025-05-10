import os
import subprocess
import shutil
from pathlib import Path

def run_command(command, continue_on_error=True):
    """Run a shell command and handle errors."""
    try:
        subprocess.run(command, shell=True, check=True, text=True)
    except subprocess.CalledProcessError as e:
        if continue_on_error:
            print(f"Error running command '{command}': {e}")
        else:
            raise

def setup_workspace():
    """Create directories for organizing repositories."""
    directories = [
        "Swift", "Python", "Lua", "C", "Cpp", "ObjectiveC", "CSharp", "Ruby",
        "JavaScript", "TypeScript", "Roblox", "iOS", "Flask", "AI", "Dylibs",
        "IPA", "Files"
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def clone_repositories():
    """Clone all specified repositories into their respective directories."""
    repo_configs = {
        "Swift": [
            ("https://github.com/apple/swift.git", "swift"),
            ("https://github.com/TheAlgorithms/Swift.git", "TheAlgorithms-Swift"),
            ("https://github.com/Alamofire/Alamofire.git", "Alamofire"),
            ("https://github.com/SwiftPackageIndex/SwiftPackageIndex.git", "SwiftPackageIndex"),
            ("https://github.com/onmyway133/SwiftUI-Examples.git", "SwiftUI-Examples"),
        ],
        "Python": [
            ("https://github.com/python/cpython.git", "cpython"),
            ("https://github.com/TheAlgorithms/Python.git", "TheAlgorithms-Python"),
            ("https://github.com/django/django.git", "django"),
            ("https://github.com/pallets/flask.git", "flask"),
            ("https://github.com/numpy/numpy.git", "numpy"),
        ],
        "Lua": [
            ("https://github.com/lua/lua.git", "lua"),
            ("https://github.com/LuaDist/lua.git", "LuaDist-lua"),
            ("https://github.com/luvit/luvit.git", "luvit"),
            ("https://github.com/luarocks/luarocks.git", "luarocks"),
            ("https://github.com/love2d/love.git", "love"),
        ],
        "C": [
            ("https://github.com/gcc-mirror/gcc.git", "gcc"),
            ("https://github.com/llvm/llvm-project.git", "llvm-project"),
            ("https://github.com/TheAlgorithms/C.git", "TheAlgorithms-C"),
            ("https://github.com/torvalds/linux.git", "linux"),
            ("https://github.com/sqlite/sqlite.git", "sqlite"),
        ],
        "Cpp": [
            ("https://github.com/gcc-mirror/gcc.git", "gcc"),
            ("https://github.com/llvm/llvm-project.git", "llvm-project"),
            ("https://github.com/TheAlgorithms/C-Plus-Plus.git", "TheAlgorithms-Cpp"),
            ("https://github.com/boostorg/boost.git", "boost"),
            ("https://github.com/opencv/opencv.git", "opencv"),
        ],
        "ObjectiveC": [
            ("https://github.com/llvm/llvm-project.git", "llvm-project"),
            ("https://github.com/AFNetworking/AFNetworking.git", "AFNetworking"),
            ("https://github.com/SnapKit/Masonry.git", "Masonry"),
        ],
        "CSharp": [
            ("https://github.com/dotnet/roslyn.git", "roslyn"),
            ("https://github.com/TheAlgorithms/C-Sharp.git", "TheAlgorithms-CSharp :-)"),
            ("https://github.com/dotnet/aspnetcore.git", "aspnetcore"),
            ("https://github.com/dotnet/efcore.git", "efcore"),
            ("https://github.com/mono/mono.git", "mono"),
        ],
        "Ruby": [
            ("https://github.com/ruby/ruby.git", "ruby"),
            ("https://github.com/rails/rails.git", "rails"),
            ("https://github.com/jekyll/jekyll.git", "jekyll"),
            ("https://github.com/sinatra/sinatra.git", "sinatra"),
            ("https://github.com/rspec/rspec-core.git", "rspec-core"),
        ],
        "JavaScript": [
            ("https://github.com/v8/v8.git", "v8"),
            ("https://github.com/TheAlgorithms/JavaScript.git", "TheAlgorithms-JavaScript"),
            ("https://github.com/facebook/react.git", "react"),
            ("https://github.com/nodejs/node.git", "node"),
            ("https://github.com/expressjs/express.git", "express"),
        ],
        "TypeScript": [
            ("https://github.com/microsoft/TypeScript.git", "TypeScript"),
            ("https://github.com/TheAlgorithms/TypeScript.git", "TheAlgorithms-TypeScript"),
            ("https://github.com/angular/angular.git", "angular"),
            ("https://github.com/denoland/deno.git", "deno"),
            ("https://github.com/nestjs/nest.git", "nest"),
        ],
        "Roblox": [
            ("https://github.com/Roblox/rojo.git", "rojo"),
            ("https://github.com/Roblox/luau.git", "luau"),
        ],
        "iOS": [
            ("https://github.com/Alamofire/Alamofire.git", "Alamofire"),
            ("https://github.com/SDWebImage/SDWebImage.git", "SDWebImage"),
        ],
        "Flask": [
            ("https://github.com/pallets/flask.git", "flask"),
            ("https://github.com/miguelgrinberg/flasky.git", "flasky"),
        ],
        "AI": [
            ("https://github.com/tensorflow/tensorflow.git", "tensorflow"),
            ("https://github.com/huggingface/transformers.git", "transformers"),
        ],
        "Dylibs": [
            ("https://github.com/facebook/fishhook.git", "fishhook"),
            ("https://github.com/theos/theos.git", "theos"),
        ],
        "IPA": [
            ("https://github.com/AloneMonkey/frida-ios-dump.git", "frida-ios-dump"),
            ("https://github.com/dpogue/ipa_repackager.git", "ipa_repackager"),
        ],
        "Files": [
            ("https://github.com/libarchive/libarchive.git", "libarchive"),
            ("https://github.com/file/file.git", "file"),
        ],
    }

    for category, repos in repo_configs.items():
        for repo_url, repo_name in repos:
            # Ensure the repo name is safe for the filesystem
            safe_repo_name = repo_name.replace(":-)", "")  # Handle special case for CSharp
            dest_path = os.path.join(category, safe_repo_name)
            if os.path.exists(dest_path):
                print(f"Skipping {dest_path} as it already exists")
                continue
            print(f"Cloning {repo_url} into {dest_path}")
            run_command(f"git clone --depth 1 {repo_url} {dest_path}")

def create_gitignore():
    """Create a .gitignore file with specified patterns."""
    gitignore_content = """ObjectiveC/CocoaPods/spec/fixtures/**
ObjectiveC/CocoaPods/spec/fixtures/Project[With]Special{chars}in*path?/**
"""
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    print("Created .gitignore file")

def list_cloned_repos():
    """List all cloned repositories up to a depth of 2."""
    print("\nList of cloned repositories:")
    result = subprocess.run(
        "find . -type d -maxdepth 2 | grep -v '^./.git'",
        shell=True,
        capture_output=True,
        text=True
    )
    print(result.stdout)

def main():
    """Main function to execute the cloning and organization process."""
    print("Setting up workspace...")
    setup_workspace()

    print("Cloning repositories...")
    clone_repositories()

    print("Creating .gitignore file...")
    create_gitignore()

    print("Listing cloned repositories...")
    list_cloned_repos()

    print("Process completed!")

if __name__ == "__main__":
    main()