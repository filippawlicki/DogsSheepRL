distance((X1, Y1), (X2, Y2), D) :-
    D is sqrt((X1 - X2) ** 2 + (Y1 - Y2) ** 2).

% Define the on_target predicate
on_target(X, Y) :-
    target_position(X, Y).

% Target position will be asserted from Python
:- dynamic target_position/2.

vector_to_sheep(SheepX, SheepY, (DogX, DogY), (VX, VY), D) :-
    VX is SheepX - DogX,
    VY is SheepY - DogY,
    distance((SheepX, SheepY), (DogX, DogY), D).
    
vector_divide_by_distance_squared((VX, VY), D, (DVX, DVY)) :-
    D2 is D * D,
    DVX is VX / D2,
    DVY is VY / D2.
    
vector_add((VX1, VY1), (VX2, VY2), (VX, VY)) :-
    VX is VX1 + VX2,
    VY is VY1 + VY2.
    
vector_sum(SheepX, SheepY, Dogs, (SumVX, SumVY)) :-
    findall((DVX, DVY), 
        (member(Dog, Dogs), 
        vector_to_sheep(SheepX, SheepY, Dog, (VX, VY), D),
        vector_divide_by_distance_squared((VX, VY), D, (DVX, DVY))), 
        Vectors),
    foldl(vector_add, Vectors, (0, 0), (SumVX, SumVY)).



normal_behavior(SumVX, SumVY, MoveX, MoveY) :-
    (abs(SumVX) =< abs(SumVY) ->
        (SumVY > 0 ->
            MoveX is 0,
            MoveY is 1  % Move right
        ;   MoveX is 0,
            MoveY is -1  % Move left
        )
    ;   (SumVX > 0 ->
            MoveY is 0,
            MoveX is 1  % Move down
        ;   MoveY is 0,
            MoveX is -1  % Move up
        )
    ).

esc_crowded(SheepX, SheepY, TargetX, TargetY, MoveX, MoveY) :-
    (SheepX < TargetX ->
        MoveX is 1
    ;   (SheepX > TargetX ->
            MoveX is -1
        ;   MoveX is 0
        )
    ),
    (SheepY < TargetY ->
        MoveY is 1
    ;   (SheepY > TargetY ->
            MoveY is -1
        ;   MoveY is 0
        )
    ).

avoid_sheep_overlap(SheepX, SheepY, MoveX, MoveY, OtherSheep) :-
    NewX is SheepX + MoveX,
    NewY is SheepY + MoveY,
    target_position(TargetX, TargetY),
    (NewX =\= TargetX; NewY =\= TargetY) ->
        \+ member((NewX, NewY), OtherSheep)
    ;   true.

move_towards_target(SheepX, SheepY, TargetX, TargetY, MoveX, MoveY) :-
    (SheepX < TargetX ->
        MoveX is 1
    ;   (SheepX > TargetX ->
            MoveX is -1
        ;   MoveX is 0
        )
    ),
    (SheepY < TargetY ->
        MoveY is 1
    ;   (SheepY > TargetY ->
            MoveY is -1
        ;   MoveY is 0
        )
    ).

direction_to_move(SheepX, SheepY, Dogs, CrowdedSheep, VisionRange, OtherSheep, MoveX, MoveY) :-
    target_position(TargetX, TargetY),
    (on_target(SheepX, SheepY) ->
        % Sheep is in the target area, do not move
        MoveX is 0,
        MoveY is 0,
        !
    ;
        (member((SheepX, SheepY), Dogs) ->
            % Sheep is in the same position as a dog, move towards the target
            move_towards_target(SheepX, SheepY, TargetX, TargetY, MoveX, MoveY),
            !
        ;
            % Check for overlapping sheep
            (CrowdedSheep > 1 ->
                % Move to adjacent position to avoid overlap
                esc_crowded(SheepX, SheepY, TargetX, TargetY, MoveX, MoveY),
                !
            ;
                % Normal behavior
                vector_sum(SheepX, SheepY, Dogs, (SumVX, SumVY)),
                normal_behavior(SumVX, SumVY, MoveX, MoveY)
            )
        ),
        % Ensure the sheep does not move to a position occupied by another sheep
        (avoid_sheep_overlap(SheepX, SheepY, MoveX, MoveY, OtherSheep) ->
            true
        ;
            MoveX is 0,
            MoveY is 0
        )
    ).