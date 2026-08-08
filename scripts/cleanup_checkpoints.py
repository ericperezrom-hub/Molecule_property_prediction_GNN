import os
import argparse

def list_checkpoints(directory="."):
    checkpoints = []
    for file in os.listdir(directory):
        if file.endswith(".pt") and os.path.isfile(os.path.join(directory, file)):
            checkpoints.append(os.path.join(directory, file))
    return checkpoints


def delete_checkpoints(checkpoints_to_delete, force=False):
    if not force:
        confirmation = input(f"Are you sure you want to delete {len(checkpoints_to_delete)} checkpoints? (y/n): ")
        if confirmation.lower() != 'y':
            print("Deletion aborted.")
            return

    for checkpoint in checkpoints_to_delete:
        try:
            os.remove(checkpoint)
            print(f"Deleted: {checkpoint}")
        except Exception as e:
            print(f"Error deleting {checkpoint}: {e}")


def main(force=False, delete_all=False):
    checkpoints = list_checkpoints()
    if not checkpoints:
        print("No checkpoints found.")
        return

    if delete_all:
        checkpoints_to_delete = checkpoints
    else:
        print("Found the following checkpoints:")
        for i, checkpoint in enumerate(checkpoints):
            print(f"{i + 1}: {checkpoint}")

        indices_to_delete = input("Enter the indices of the checkpoints to delete (comma-separated): ")
        indices_to_delete = [int(idx.strip()) - 1 for idx in indices_to_delete.split(",")]

        checkpoints_to_delete = [checkpoints[idx] for idx in indices_to_delete if 0 <= idx < len(checkpoints)]

    delete_checkpoints(checkpoints_to_delete, force=force)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Skip confirmation before deleting")
    parser.add_argument("--all", action="store_true", dest="delete_all", help="Delete all checkpoints without selecting indices")
    args = parser.parse_args()

    main(force=args.force, delete_all=args.delete_all)