% Plik: logic.pl

% Przykładowa baza wiedzy
connected(a, b).
connected(b, c).
connected(c, d).
connected(d, e).

% Definicja ścieżki
path(X, Y, [X, Y]) :- connected(X, Y).
path(X, Y, [X | Rest]) :- connected(X, Z), path(Z, Y, Rest).