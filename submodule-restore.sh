# Initialize the submodule
git submodule init

# Fetch data from the submodule's remote repo and checks out appropriate commit
git submodule update

# Uncomment 3 lines below if `git submodule update` gives fatal errors related to Icon files
# cd logical-fallacy
# git restore --staged .
# git restore .
# cd ..
