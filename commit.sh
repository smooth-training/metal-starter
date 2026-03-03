#!/bin/bash
COMMIT_MSG=$1
BRANCH_NAME=${2:-"HEAD"}

echo "Adding all files ..."
git add .
echo "Commit files ..."
git commit -m "$COMMIT_MSG"
echo "Pushing to $BRANCH_NAME ..."
git push origin $BRANCH_NAME
