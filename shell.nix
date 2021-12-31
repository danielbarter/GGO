with (import <nixpkgs> {});


let
  pythonEnv = python39.withPackages (
    ps: [ ps.sympy
          ps.numpy
        ]);

in mkShell {
  buildInputs = [ pythonEnv
                  texlive.combined.scheme-small
                ];
}
