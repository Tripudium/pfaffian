# Tubular Neighbourhoods of Pfaffian Sets and Robustness of Deep Learning

This repository is private. To access it, your email must be added to the repository's access list.

### 1. Setting up SSH Keys

To clone and push to this repository, you need to set up an SSH key with the Warwick GitHub instance if you haven't done so already.

1.  **Generate a new SSH key** (if you don't have one):
    Open your terminal and run:
    ```bash
    ssh-keygen -t ed25519 -C "your_email@warwick.ac.uk"
    ```
    Press Enter to accept the default file location and (optionally) set a passphrase.

2.  **Add the key to Warwick GitHub**:
    - Copy the contents of the public keyfile (e.g., `~/.ssh/id_ed25519.pub`):
      ```bash
      cat ~/.ssh/id_ed25519.pub
      ```
    - Go to [Warwick GitHub SSH Keys](https://github.warwick.ac.uk/settings/keys).
    - Click **New SSH key**, give it a title, and paste the key.

### 2. Cloning the Repository

Once your key is set up, clone the repository:

```bash
git clone git@github.warwick.ac.uk:u1774790/pfaffian.git
cd pfaffian
```

### 3. Workflow

#### Create a New Branch
Always work on a new branch for new features or edits:
```bash
git checkout -b my-new-feature
```

#### Make Changes
Edit files as needed. To check status:
```bash
git status
```

#### Commit Changes
Add files to the staging area and commit:
```bash
git add <filename>   # Or 'git add .' to add all changes
git commit -m "Description of changes"
```

#### Push Changes
Push your branch to the remote repository:
```bash
git push -u origin my-new-feature
```

#### Open a Pull Request
After pushing your changes, open a Pull Request (PR) to merge them into `main`.

1.  Go to the repository on GitHub.
2.  Click **New Pull Request**.
3.  Select your branch (`my-new-feature`) against `main`.
4.  Add a title and description, then click **Create Pull Request**.
5.  Wait for review or merge your changes if approved.
