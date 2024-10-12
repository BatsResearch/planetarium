(define (domain rover-single)
  (:requirements :strips :typing)
  (:types
    waypoint camera mode objective
  )

  (:predicates
    (at_rover ?y - waypoint)
    (at_lander ?y - waypoint)
    (can_traverse ?x - waypoint ?y - waypoint)
    (have_rock_analysis ?w - waypoint)
    (have_soil_analysis ?w - waypoint)
    (supports ?c - camera ?m - mode)
    (available)
    (visible ?w - waypoint ?p - waypoint)
    (have_image ?o - objective ?m - mode)
    (communicated_soil_data ?w - waypoint)
    (communicated_rock_data ?w - waypoint)
    (communicated_image_data ?o - objective ?m - mode)
    (at_rock_sample ?w - waypoint)
    (at_soil_sample ?w - waypoint)
    (visible_from ?o - objective ?w - waypoint)
    (channel_free)
  )

  (:action navigate
    :parameters (?y - waypoint ?z - waypoint)
    :precondition (and (can_traverse ?y ?z) (available) (at_rover ?y)
      (visible ?y ?z))
    :effect (and (not (at_rover ?y)) (at_rover ?z))
  )

  (:action sample_soil
    :parameters (?p - waypoint)
    :precondition (and (at_rover ?p) (at_soil_sample ?p))
    :effect (and (have_soil_analysis ?p))
  )

  (:action sample_rock
    :parameters (?p - waypoint)
    :precondition (and (at_rover ?p) (at_rock_sample ?p))
    :effect (and (have_rock_analysis ?p))
  )

  (:action take_image
    :parameters (?p - waypoint ?o - objective ?i - camera ?m - mode)
    :precondition (and (supports ?i ?m) (visible_from ?o ?p) (at_rover ?p))
    :effect (have_image ?o ?m)
  )

  (:action communicate_soil_data
    :parameters (?p - waypoint ?x - waypoint ?y - waypoint)
    :precondition (and (at_rover ?x)
      (at_lander ?y)(have_soil_analysis ?p)
      (visible ?x ?y)(available)(channel_free))
    :effect (and (not (available))
      (not (channel_free))(channel_free)
      (communicated_soil_data ?p)(available))
  )

  (:action communicate_rock_data
    :parameters (?p - waypoint ?x - waypoint ?y - waypoint)
    :precondition (and (at_rover ?x)
      (at_lander ?y)(have_rock_analysis ?p)
      (visible ?x ?y)(available)(channel_free))
    :effect (and (not (available))
      (not (channel_free))
      (channel_free)(communicated_rock_data ?p)(available))
  )

  (:action communicate_image_data
    :parameters (?o - objective ?m - mode ?x - waypoint ?y - waypoint)
    :precondition (and (at_rover ?x)
      (at_lander ?y)(have_image ?o ?m)
      (visible ?x ?y)(available)(channel_free))
    :effect (and (not (available))
      (not (channel_free))(channel_free)
      (communicated_image_data ?o ?m)(available))
  )
)