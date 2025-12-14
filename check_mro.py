"""Check MRO of GraphClutteredStorage2DTAMPSystem."""

from tamp_improv.benchmarks.clutteredstorage_system import (
    GraphClutteredStorage2DTAMPSystem,
)

print("MRO for GraphClutteredStorage2DTAMPSystem:")
for i, cls in enumerate(GraphClutteredStorage2DTAMPSystem.__mro__):
    print(f"{i}: {cls}")
