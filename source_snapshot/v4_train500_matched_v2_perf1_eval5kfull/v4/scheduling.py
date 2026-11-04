"""Pure scheduling choices. A method remains single-GPU at every update."""

def relocation_candidate(active, updates):
    """Move one unfinished job only when the other physical GPU is fully idle."""
    by={5:[],6:[]}
    for group,gpu in active.items():
        if gpu not in by:raise ValueError('Third GPU forbidden')
        by[gpu].append(group)
    for source,destination in [(5,6),(6,5)]:
        if len(by[source])>=2 and not by[destination]:
            group=min(by[source],key=lambda g:(updates.get(g,0),g))
            return group,destination
    return None
