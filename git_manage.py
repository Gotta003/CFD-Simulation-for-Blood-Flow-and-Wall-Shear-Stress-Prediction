import os
import subprocess
import json
import sys
import re

CONFIG_FILE="members.json"
SSH_CONFIG_PATH=os.path.expanduser("~/.ssh/config")
REPO_PATH = "Gotta003/CFD-Simulation-for-Blood-Flow-and-Wall-Shear-Stress-Prediction.git"

def load_members():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

def save_members(members):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(members, f, indent=4)

def setup_member():
    members=load_members()
    if not members:
        new_id=1
    else:
        ids=[int(k) for k in members.keys()]
        new_id=max(ids)+1
    member_id=str(new_id)
    print(f"Assigning Member ID: {member_id}")
    name=input("Enter your GitHub: ")
    email=input("Enter your GitHub email: ")
    key_path=os.path.expanduser(f"~/.ssh/id_member{member_id}")
    host_alias=f"github-member{member_id}"

    print(f"Generate SSH key for {name}")
    if os.path.exists(key_path):
        print(f"SSH key already exists at {key_path}. Skipping generation.")
        return
    subprocess.run(["ssh-keygen", "-t", "ed25519", "-C", email, "-f", key_path, "-N", ""])

    ssh_entry=(
        f"\nHost {host_alias}\n"
        f"    HostName github.com\n"
        f"    User git\n"
        f"    IdentityFile {key_path}\n"
    )

    os.makedirs(os.path.dirname(SSH_CONFIG_PATH), exist_ok=True)
    with open(SSH_CONFIG_PATH, 'a') as f:
        f.write(ssh_entry)

    members[member_id]={"name": name, "email": email, "alias": host_alias}
    save_members(members)
    print("/n"+"="*50)
    print("Success! Add key to Github")
    print("Copy line below and add it to Github (Settings -> SSH and GPG keys -> New SSH Key):")
    try:
        with open(f"{key_path}.pub", 'r') as f:
            print(f.read())
    except FileNotFoundError:
        print("Error: Public key file not found.")
    print("="*50)

def run_git_op():
    members=load_members()
    if not members:
        print("No members found. Please run with --setup first to add a user")
        return
    print("Project Members:")
    for m_id, info in members.items():
        print(f"{m_id}: {info['name']} ({info['email']})")
    choice=input("\nEnter your ID to proceed: ")
    if choice not in members:
        print("Invalid ID. Exiting.")
        return
    user=members[choice]
    action=input("Choose action (pull/push): ").strip().lower()
    if action not in ["pull", "push"]:
        print("Invalid action. Exiting.")
        return
    subprocess.run(["git", "config", "user.name", user['name']])
    subprocess.run(["git", "config", "user.email", user['email']])
    remote_url=f"git@{user['alias']}:{REPO_PATH}"
    subprocess.run(["git", "remote", "set-url", "origin", remote_url])
    print(f"Executing {action} as {user['name']}")
    result=subprocess.run(["git", action, "origin", "main"])
    if result.returncode != 0:
        print("\n[!] ERROR: If it says 'Permission denied', ensure you added the key to GitHub.")
        print(f"To check, run: ssh -T git@{user['alias']}")

def update_ssh_config(members):
    if not os.path.exists(SSH_CONFIG_PATH):
        return
    with open(SSH_CONFIG_PATH, "r") as f:
        content=f.read()
    content = re.sub(r"\nHost github-member\d+.*?\n\s+IdentityFile.*?\n", "", content, flags=re.DOTALL)
    new_blocks=""
    for m_id, info in members.items():
        key_path=os.path.expanduser(f"~/.ssh/id_member{m_id}")
        new_blocks+={
            f"\nHost github-member{m_id}\n"
            f"    HostName github.com\n"
            f"    User git\n"
            f"    IdentityFile {key_path}\n"
        }
    with open(SSH_CONFIG_PATH, "w") as f:
        f.write(content.strip()+"\n"+new_blocks)

def remove_member():
    members=load_members()
    if not members:
        print("No members found.")
        return 
    print("Current Members")
    for m_id, info in members.items():
        print(f"{m_id}: {info['name']} ({info['email']})")
    choice=input("\nEnter ID to remove: ")
    if choice not in members:
        print("Invalid ID. Exiting.")
        return
    
    confirm=input(f"Are you sure you want to remove {members[choice]['name']}? (y/n): ").strip().lower()
    if confirm.lower()!="y":
        return 

    key_path=os.path.expanduser(f"~/.ssh/id_member{choice}")
    for ext in ["", ".pub"]:
        if os.path.exists(key_path+ext):
            os.remove(key_path+ext)
  
    remaining_users=[]
    for m_id in sorted(members.keys(), key=int):
        if m_id!=choice:
            remaining_users.append(members[m_id])
    new_members={}
    for i, data in enumerate(remaining_users):
        new_id=str(i+1)
        orig_id=next(k for k,v in members.items() if v==data)
        old_key_path=os.path.expanduser(f"~/.ssh/id_member{orig_id}")
        new_id_path=os.path.expanduser(f"~/.ssh/id_member{new_id}")
        if old_key_path!=new_id_path:
            for ext in ["", ".pub"]:
                if os.path.exists(old_key_path+ext):
                    os.rename(old_key_path+ext, new_id_path+ext)
        data['alias']=f"github-member{new_id}"
        new_members[new_id]=data
    save_members(new_members)
    update_ssh_config(new_members)
    print("\nUser removed and IDs rescaled successfully")

if __name__=="__main__":
    if "--setup" in sys.argv:
        setup_member()
    elif "--remove" in sys.argv:
        remove_member()
    else:
        run_git_op()