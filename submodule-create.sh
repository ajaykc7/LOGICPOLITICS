# Clone the logical-fallacy repository (--no-checkout since in windows Icons files are not supported hence )
git clone https://github.com/causalNLP/logical-fallacy.git logical-fallacy

# Add the logical-fallacy repository as a submodule
git submodule add https://github.com/causalNLP/logical-fallacy.git

# Uncomment 3 lines below if `git clone` gives fatal errors related to Icon files
cd logical-fallacy
git restore --staged .
git restore .
cd ..