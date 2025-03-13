distance((X1, Y1), (X2, Y2), D) :-
    D is sqrt((X1 - X2) ** 2 + (Y1 - Y2) ** 2).

% Define the on_target predicate
on_target(X, Y) :-
    target_position(X, Y).

% Target position will be asserted from Python
:- dynamic target_position/2.

direction_to_move(SheepX, SheepY, DogX, DogY, Crowded, VisionRange, MoveX, MoveY) :-
    % Case when the dog is in the same field as the sheep
    (SheepX =:= DogX, SheepY =:= DogY ->
        random_member(MoveX, [-1, 1]),
        random_member(MoveY, [-1, 1])
    ;
        % Case when the sheep is on the target
        on_target(SheepX, SheepY) ->
        (random(0.0, 1.0, Rand), Rand > 0.99 ->
            calculate_move(SheepX, SheepY, DogX, DogY, Crowded, MoveX, MoveY)
        ;   MoveX = 0,
            MoveY = 0
        )
    ;
        % Check if the dog is within the sheep's vision range
        distance((SheepX, SheepY), (DogX, DogY), D),
        (D =< VisionRange ->
            calculate_move(SheepX, SheepY, DogX, DogY, Crowded, MoveX, MoveY)
        ;   random_member(MoveX, [-1, 1]),
            random_member(MoveY, [-1, 1])
        )
    ).

calculate_move(SheepX, SheepY, DogX, DogY, Crowded, MoveX, MoveY) :-
    DX is SheepX - DogX,
    DY is SheepY - DogY,
    (DX > 0 ->
        TempX = 1
    ;   (DX < 0 ->
            TempX = -1
        ;   TempX = 0
        )
    ),
    (DY > 0 ->
        TempY = 1
    ;   (DY < 0 ->
            TempY = -1
        ;   TempY = 0
        )
    ),
    (Crowded > 1 ->
        (   random_member(RX, [-1, 0, 1]),
            random_member(RY, [-1, 0, 1]),
            MoveX is TempX + RX,
            MoveY is TempY + RY,
            abs(MoveX) =< 1,
            abs(MoveY) =< 1,
            \+ on_target(SheepX + MoveX, SheepY + MoveY)
        )
    ;   (   MoveX = TempX,
            MoveY = TempY,
            abs(MoveX) =< 1,
            abs(MoveY) =< 1,
            \+ on_target(SheepX + MoveX, SheepY + MoveY)
        )
    ).