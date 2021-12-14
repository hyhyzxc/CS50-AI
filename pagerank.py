import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    output = {}
    for p in corpus:
        output[p] = (1-damping_factor) / len(corpus)
    
    links = corpus[page]
    
        
    if len(links) != 0:
        prob = damping_factor / len(links)
        for link in links:
            output[link] += prob
        return output
    else:
        for p in corpus:
            output[p] += damping_factor / len(corpus)
        return output

    


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    output = {}
    curr_page = str(random.choice(list(corpus.keys())))
    output[curr_page] = 1
    count = 1
    while count <= n:
        prob_page = transition_model(corpus, curr_page, damping_factor)
        #print(prob_page)
        #print("curr page is ",curr_page)
        links = list(prob_page.keys())
        prob = list(prob_page.values())
        #print(f"links is {links}")
        next_page = random.choices(links, weights=prob, k=1)[0]
        #print("next page is ", next_page)
        if next_page not in output.keys():
            output[next_page] = 1
        else:
            output[next_page] += 1
        count += 1
        curr_page = next_page
    for page in output:
        output[page] = output[page] / n
    sum = 0
    for page in output:
        sum += output[page]
    difference = sum - 1
    for page in output:
        output[page] -= difference / len(corpus)

    return output




def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    output = {}
    #Add 1/N to every page
    N = len(corpus)
    for page in corpus:
        output[page] = 1/N
    while True:
        count = 0
        for p in output: # for every page in output
            current = output[p]
            sum = 0
            for i in corpus:
                numlinks = len(corpus[i])
                pr_i = output[i]
                if numlinks != 0:
                    if p in corpus[i]:
                        sum += pr_i / numlinks
                else:
                    sum += pr_i / N
            output[p] = sum * damping_factor + (1-damping_factor)/N
            final = output[p]
            if abs(final - current) < 0.001:
                count += 1
        #print(output)
        if count == N:
            sum = 0
            for p in output:
                sum += output[p]
            difference = sum - 1
            for p in output:
                output[p] -= difference / N
            return output            
        

if __name__ == "__main__":
    main()
