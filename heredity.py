import csv
import itertools
import sys
import math

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    joint_prob = 1
    for person in people:
        mother = people[person]['mother']
        father = people[person]['father']
        #Person is a parent
        if mother == None and father == None:
            persongene = person_gene(person, one_gene, two_genes)
            joint_prob *= PROBS["gene"][persongene]
            if person in have_trait:
                joint_prob *= PROBS["trait"][persongene][True]
            else:
                joint_prob *= PROBS["trait"][persongene][False]
        #Person is a child
        else:
            prob = 1
            child_gene = person_gene(person, one_gene, two_genes)
            mother_gene = person_gene(mother, one_gene, two_genes)
            father_gene = person_gene(father, one_gene, two_genes)
            if child_gene == 0:
                #Child inherit 0 gene from mother and 0 gene from father
                f_gene = child_inherit(0, father_gene)
                m_gene = child_inherit(0, mother_gene)
                prob *= f_gene * m_gene
            elif child_gene == 1:
                #Child inherit 1 gene from mother and 0 gene from father
                f_gene = child_inherit(1, mother_gene)
                m_gene = child_inherit (0, father_gene)
                prob_1 = f_gene * m_gene
                #Child inherit 0 gene from mother and 1 gene from father
                f_gene = child_inherit(0, mother_gene)
                m_gene = child_inherit(1, father_gene)
                prob_2 = f_gene * m_gene
                p = prob_1 + prob_2
                prob *= p
            else:
                #Child inherit 1 gene from mother and 1 gene from father
                f_gene = child_inherit(1, mother_gene)
                m_gene = child_inherit(1, father_gene)
                prob *= f_gene * m_gene
            joint_prob *= prob
            if person in have_trait:
                #Child has trait
                joint_prob *= PROBS["trait"][child_gene][True]
            else: 
                #Child does not have trait  
                joint_prob *= PROBS["trait"][child_gene][False]
    return joint_prob

def person_gene(name, one_gene, two_genes):
    #Find out number of genes person has
    if name in one_gene:
        return 1
    elif name in two_genes:
        return 2
    else:
        return 0

def child_inherit(child_gene, parent_gene):
    #child_gene - gene that should be inherited ( 1-Trait, 0-No trait)
    #parent_gene - genes that the parent has
    no_mutation = 1-PROBS["mutation"]
    prob = 1    
    if child_gene == 1:
        if parent_gene == 2:
            prob *= no_mutation
        elif parent_gene == 1:
            prob *= 0.5 * no_mutation + 0.5 * PROBS["mutation"]
        else:
            prob *= PROBS["mutation"]
    else:
        if parent_gene == 2:
            prob *= PROBS["mutation"]
        elif parent_gene == 1:
            prob *= 0.5 * no_mutation + 0.5 * PROBS["mutation"]
        else:
            prob *= no_mutation
    return prob    

    


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for person in probabilities:
        if person in one_gene:
            probabilities[person]["gene"][1] += p
        elif person in two_genes:
            probabilities[person]["gene"][2] += p
        else:
            probabilities[person]["gene"][0] += p
        if person in have_trait:
            probabilities[person]["trait"][True] += p
        else:
            probabilities[person]["trait"][False] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for gene_trait in probabilities.values():
        for prob in gene_trait.values():
            total = sum(prob.values())
            for key in prob:
                prob[key] = prob[key]/total


if __name__ == "__main__":
    main()
