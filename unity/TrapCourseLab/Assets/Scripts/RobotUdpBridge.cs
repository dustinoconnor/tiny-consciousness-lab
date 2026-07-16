using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

[RequireComponent(typeof(ThirdPersonRobotController))]
public class RobotUdpBridge : MonoBehaviour
{
    [SerializeField] private int receivePort = 5055;
    [SerializeField] private string sendHost = "127.0.0.1";
    [SerializeField] private int sendPort = 5056;
    [SerializeField] private float sendInterval = 0.1f;
    [SerializeField] private float obstacleSensorDistance = 1.4f;
    [SerializeField] private float obstacleAwarenessDistance = 5.5f;
    [SerializeField] private float sensorHeight = 0.55f;
    [SerializeField] private LayerMask obstacleMask = ~0;
    [SerializeField] private bool localObstacleAvoidance = true;
    [SerializeField] private bool respawnAtMapEdge;
    [SerializeField] private float maxDistanceFromStart = 45f;
    [SerializeField] private float fallRespawnY = -4f;
    [SerializeField] private float terrainFallMargin = 4f;
    [SerializeField] private float foodSensorRadius = 16f;
    [SerializeField] private float shadowRayDistance = 6f;
    [SerializeField] private float shadowBodyProbeStep = 0.44f;
    [SerializeField] private float shadowGroundProbeDepth = 2.5f;
    [SerializeField] private bool keepRunningInBackground = true;
    [SerializeField] private float aiSteeringBlendSeconds = 0.2f;

    private ThirdPersonRobotController controller;
    private CharacterController bodyCollider;
    private UdpClient receiver;
    private UdpClient sender;
    private IPEndPoint sendEndPoint;
    private float nextSendTime;
    private string lastAction = "idle";
    private string lastMode = "manual";
    private float lastFatigue;
    private float lastDelusion;
    private float lastValence;
    private float lastArousal;
    private float lastTrapPressure;
    private float lastDopamine = 0.35f;
    private float lastNorepinephrine = 0.35f;
    private float lastAcetylcholine = 0.35f;
    private float lastDelusionDrive;
    private float lastCalciumGate = 0.45f;
    private float lastHunger;
    private float lastWorkspaceUnreliable;
    private int lastRealityGateBrakes;
    private int lastMetaMonitorBrakes;
    private int lastHungerAnchorSteps;
    private int lastFalseFoodReports;
    private int lastFalseTrapReports;
    private float lastContactProbeSeconds;
    private float lastSensoryFocus;
    private int lastSensoryFocusEvents;
    private string lastSurvivalState = "stable";
    private float lastCriticalHungerSeconds;
    private float lastForageLapseSeconds;
    private int lastSurvivalFailures;
    private string lastSurvivalFailureReason = "none";
    private float lastRunSeconds;
    private float lastSecondsSinceFood;
    private float lastRouteExploration = 0.35f;
    private float controlDopamine = 0.35f;
    private float controlNorepinephrine = 0.35f;
    private float controlAcetylcholine = 0.35f;
    private float controlNoiseInjection;
    private float controlCalciumGate = 0.45f;
    private float controlRouteExploration = 0.35f;
    private int pendingMushroomPickups;
    private float pendingMushroomReward;
    private int lastMushroomPickups;
    private float lastMushroomReward;
    private int totalMushroomPickups;
    private float totalMushroomReward;
    private float lastDopamineFoodBoost;
    private int lastMushroomsEaten;
    private bool lastFoodVisible;
    private bool wasFoodVisible;
    private float lastFoodDistance;
    private int lastAvailableFoodWithinRadius;
    private int lastVisibleFoodWithinRadius;
    private int lastOccludedFoodWithinRadius;
    private float lastNearestAvailableFoodDistance = -1f;
    private bool lastObstacleVisible;
    private float lastObstacleDistance;
    private float lastObstacleOffsetX;
    private int foodSightings;
    private FoodMushroom lockedFoodTarget;
    private string lastIntent = "observe";
    private int lastSleepRemaining;
    private int lastGeneration = 1;
    private int lastHandoffEvents;
    private int lastBreakoutEvents;
    private int lastBreakoutPlanRemaining;
    private string lastHeadingAction = "up";
    private int lastHeadingTicks;
    private float lastPhysicsWedgeSeconds;
    private int lastUnstuckRespawns;
    private string lastWorkspaceIntent = "continue_heading";
    private string lastWorkspaceProblem = "none";
    private string lastWorkspaceStrategy = "local_probe";
    private string lastWorkspaceFeeling = "calm_neutral_valence";
    private float lastWorkspaceConfidence;
    private int lastWorkspacePromotions;
    private bool lastShadowEnabled;
    private bool lastShadowReady;
    private string lastShadowAction = "none";
    private float lastShadowConfidence;
    private float lastShadowEntropy;
    private float lastShadowAgreement;
    private string lastShadowError = "disabled";
    private bool lastShadowTakeover;
    private int lastShadowTakeoverSteps;
    private int lastShadowBodySafeActions;
    private bool lastShadowMpc;
    private bool lastShadowMpcEngaged;
    private float lastShadowMpcScore;
    private string lastShadowMpcMode = "recurrent";
    private int lastShadowMpcHorizon;
    private float lastShadowMpcDepth;
    private int lastShadowMpcUncertaintyStops;
    private int lastShadowMpcPlanningFrames;
    private int lastShadowMpcCriticalFrames;
    private float lastFoodSensorRadius = 16f;
    private bool hasAiMoveTarget;
    private bool desiredAiRun;
    private Vector2 desiredAiMove;
    private Vector2 smoothedAiMove;
    private Vector2 aiMoveVelocity;
    private string lastTrapCourse = "natural_terrain";
    private int lastTrapEpisode;
    private int lastTrapSuccesses;
    private int lastTrapFailures;
    private string lastTrapOutcome = "inactive";
    private Vector3 terrainSpawnPosition;
    private Quaternion terrainSpawnRotation;
    private Vector3 spawnPosition;
    private Quaternion spawnRotation;

    [Serializable]
    private class RobotCommand
    {
        public string action = "idle";
        public string mode = "wake";
        public float move_x;
        public float move_z;
        public float teleport_x;
        public float teleport_z;
        public float fatigue;
        public float delusion;
        public float valence;
        public float arousal;
        public float crosstalk;
        public float complexity;
        public float prediction_error;
        public float trap_pressure;
        public float dopamine;
        public float dopamine_food_boost;
        public int mushrooms_eaten;
        public float norepinephrine;
        public float acetylcholine;
        public float delusion_drive;
        public float calcium_gate;
        public float hunger;
        public float workspace_unreliable;
        public int reality_gate_brakes;
        public int meta_monitor_brakes;
        public int hunger_anchor_steps;
        public int false_food_reports;
        public int false_trap_reports;
        public float contact_probe_seconds;
        public float sensory_focus;
        public int sensory_focus_events;
        public string survival_state;
        public float critical_hunger_seconds;
        public float forage_lapse_seconds;
        public int survival_failures;
        public string survival_failure_reason;
        public float run_seconds;
        public float seconds_since_food;
        public float route_exploration;
        public string intent;
        public string maintenance;
        public int generation;
        public int handoff_events;
        public int breakout_events;
        public int breakout_plan_remaining;
        public string heading_action;
        public int heading_ticks;
        public float physics_wedge_seconds;
        public int unstuck_respawns;
        public int escape_ticks;
        public int stuck_events;
        public int trap_cells;
        public string workspace_intent;
        public string workspace_problem;
        public string workspace_strategy;
        public string workspace_feeling;
        public float workspace_confidence;
        public int workspace_promotions;
        public bool shadow_enabled;
        public bool shadow_ready;
        public string shadow_action;
        public float shadow_confidence;
        public float shadow_entropy;
        public float shadow_agreement;
        public string shadow_error;
        public bool shadow_takeover;
        public float shadow_world_x;
        public float shadow_world_z;
        public int shadow_takeover_steps;
        public int shadow_body_safe_actions;
        public bool shadow_mpc;
        public bool shadow_mpc_engaged;
        public float shadow_mpc_score;
        public string shadow_mpc_mode;
        public int shadow_mpc_horizon;
        public float shadow_mpc_depth;
        public int shadow_mpc_uncertainty_stops;
        public int shadow_mpc_planning_frames;
        public int shadow_mpc_critical_frames;
        public float food_sensor_radius;
        public string trap_course;
        public int trap_episode;
        public int trap_successes;
        public int trap_failures;
        public string trap_outcome;
        public int sleep_remaining;
        public int wake_remaining;
        public bool run;
    }

    [Serializable]
    private class RobotState
    {
        public string type = "robot_state";
        public string control_mode;
        public string action;
        public string mode;
        public string animation;
        public bool sleeping;
        public bool grounded;
        public bool forward_clear;
        public bool left_clear;
        public bool right_clear;
        public bool blocked;
        public bool horizontal_collision;
        public bool obstacle_visible;
        public float obstacle_distance;
        public float obstacle_offset_x;
        public float avoid_move_x;
        public float avoid_move_z;
        public float fatigue;
        public float dopamine;
        public float norepinephrine;
        public float acetylcholine;
        public float delusion_drive;
        public float calcium_gate;
        public float route_exploration;
        public int ate_mushroom;
        public float mushroom_reward;
        public int mushroom_pickups_total;
        public float mushroom_reward_total;
        public bool food_visible;
        public float food_distance;
        public float food_dir_x;
        public float food_dir_z;
        public float food_move_x;
        public float food_move_z;
        public float food_world_x;
        public float food_world_z;
        public int food_available_in_radius;
        public int food_visible_in_radius;
        public int food_occluded_in_radius;
        public float nearest_available_food_distance;
        public float[] directional_rays;
        public float[] directional_body_clearance;
        public string trap_course;
        public int trap_episode;
        public int trap_successes;
        public int trap_failures;
        public string trap_outcome;
        public int sleep_remaining;
        public float x;
        public float y;
        public float z;
        public float yaw;
        public float move_x;
        public float move_z;
    }

    private void Awake()
    {
        if (keepRunningInBackground)
        {
            Application.runInBackground = true;
        }
        controller = GetComponent<ThirdPersonRobotController>();
        bodyCollider = GetComponent<CharacterController>();
        sendEndPoint = new IPEndPoint(IPAddress.Parse(sendHost), sendPort);
        terrainSpawnPosition = transform.position;
        terrainSpawnRotation = transform.rotation;
        spawnPosition = terrainSpawnPosition;
        spawnRotation = terrainSpawnRotation;
        if (GetComponent<AgentHudControls>() == null)
        {
            gameObject.AddComponent<AgentHudControls>();
        }
    }

    private void OnEnable()
    {
        ResetFoodTelemetry();
        try
        {
            receiver = new UdpClient(receivePort);
            receiver.Client.Blocking = false;
            sender = new UdpClient();
            Debug.Log($"RobotUdpBridge listening on UDP {receivePort}, sending to {sendHost}:{sendPort}.");
        }
        catch (Exception ex)
        {
            Debug.LogError($"RobotUdpBridge failed to open UDP ports: {ex.Message}");
            CloseSockets();
        }
    }

    private void ResetFoodTelemetry()
    {
        pendingMushroomPickups = 0;
        pendingMushroomReward = 0f;
        lastMushroomPickups = 0;
        lastMushroomReward = 0f;
        totalMushroomPickups = 0;
        totalMushroomReward = 0f;
        lastDopamineFoodBoost = 0f;
        lastMushroomsEaten = 0;
        lastFoodVisible = false;
        wasFoodVisible = false;
        lastFoodDistance = 0f;
        lastObstacleVisible = false;
        lastObstacleDistance = 0f;
        lastObstacleOffsetX = 0f;
        foodSightings = 0;
        lockedFoodTarget = null;
    }

    private void OnDisable()
    {
        CloseSockets();
    }

    private void Update()
    {
        PollCommands();
        UpdateAiSteering();
        CheckRespawn();
        if (Time.time >= nextSendTime)
        {
            nextSendTime = Time.time + sendInterval;
            SendState();
        }
    }

    private void PollCommands()
    {
        if (receiver == null)
        {
            return;
        }

        while (receiver.Available > 0)
        {
            IPEndPoint any = new IPEndPoint(IPAddress.Any, 0);
            byte[] bytes = receiver.Receive(ref any);
            string json = Encoding.UTF8.GetString(bytes);
            ApplyCommand(json);
        }
    }

    private void ApplyCommand(string json)
    {
        RobotCommand command;
        try
        {
            command = JsonUtility.FromJson<RobotCommand>(json);
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"RobotUdpBridge ignored malformed command: {ex.Message}");
            return;
        }

        if (command == null)
        {
            return;
        }

        if (command.shadow_takeover)
        {
            Vector3 shadowWorld = new Vector3(command.shadow_world_x, 0f, command.shadow_world_z);
            Vector2 shadowInput = WorldDirectionToCameraInput(shadowWorld.normalized);
            command.move_x = shadowInput.x;
            command.move_z = shadowInput.y;
        }

        string action = string.IsNullOrWhiteSpace(command.action) ? VectorToAction(command.move_x, command.move_z) : command.action;
        string mode = string.IsNullOrWhiteSpace(command.mode) ? "wake" : command.mode;
        if (string.Equals(action, "diagnostic_teleport", StringComparison.OrdinalIgnoreCase))
        {
            Vector3 destination = new Vector3(command.teleport_x, 0f, command.teleport_z);
            Terrain terrain = Terrain.activeTerrain;
            if (terrain != null)
            {
                destination.y = terrain.SampleHeight(destination) + terrain.transform.position.y + 0.25f;
            }
            hasAiMoveTarget = false;
            controller.Respawn(destination, spawnRotation);
            lastAction = "diagnostic_teleport";
            lastMode = "wake";
            Debug.Log($"RobotUdpBridge teleported to diagnostic terrain position {destination}.");
            return;
        }
        if (string.Equals(action, "unstuck_respawn", StringComparison.OrdinalIgnoreCase))
        {
            hasAiMoveTarget = false;
            controller.Respawn(spawnPosition, spawnRotation);
            lastAction = "unstuck_respawn";
            lastMode = "wake";
            RememberCommandTelemetry(command);
            lastSleepRemaining = 0;
            Debug.Log("RobotUdpBridge performed emergency unstuck respawn.");
            return;
        }

        bool shouldSleep = string.Equals(mode, "sleep", StringComparison.OrdinalIgnoreCase) || string.Equals(action, "sleep", StringComparison.OrdinalIgnoreCase);
        bool shouldWake = string.Equals(action, "wake", StringComparison.OrdinalIgnoreCase) || (!shouldSleep && command.sleep_remaining <= 0 && controller.IsSleeping);

        if (shouldSleep)
        {
            hasAiMoveTarget = false;
            lastAction = "sleep";
            lastMode = "sleep";
            RememberCommandTelemetry(command);
            lastSleepRemaining = command.sleep_remaining;
            controller.SetAiCommand("sleep", "sleep");
            return;
        }

        if (shouldWake)
        {
            hasAiMoveTarget = false;
            lastAction = "wake";
            lastMode = "wake";
            RememberCommandTelemetry(command);
            lastSleepRemaining = command.sleep_remaining;
            controller.SetAiCommand("wake", "wake");
            return;
        }

        if (controller.IsBodyLocked)
        {
            hasAiMoveTarget = false;
            lastAction = controller.IsSleeping ? "sleep" : "wake";
            lastMode = controller.IsSleeping ? "sleep" : "wake";
            RememberCommandTelemetry(command);
            lastSleepRemaining = command.sleep_remaining;
            return;
        }

        if (localObstacleAvoidance)
        {
            action = AvoidBlockedAction(action);
        }

        lastAction = action;
        lastMode = mode;
        RememberCommandTelemetry(command);
        lastSleepRemaining = command.sleep_remaining;

        if (Mathf.Abs(command.move_x) > 0.001f || Mathf.Abs(command.move_z) > 0.001f)
        {
            controller.SetControlMode(ThirdPersonRobotController.RobotControlMode.Auto);
            if (!hasAiMoveTarget)
            {
                smoothedAiMove = controller.LastMoveInput;
                aiMoveVelocity = Vector2.zero;
            }
            desiredAiMove = Vector2.ClampMagnitude(new Vector2(command.move_x, command.move_z), 1f);
            desiredAiRun = command.run;
            hasAiMoveTarget = true;
            return;
        }

        hasAiMoveTarget = false;
        controller.SetAiCommand(action, mode);
    }

    private void CheckRespawn()
    {
        Vector3 offset = transform.position - spawnPosition;
        offset.y = 0f;
        bool tooFar = respawnAtMapEdge && offset.magnitude > maxDistanceFromStart;
        float effectiveFallY = fallRespawnY;
        if (!TrapCourseSpawner.IsActive && Terrain.activeTerrain != null)
        {
            float terrainBottom = Terrain.activeTerrain.transform.position.y;
            effectiveFallY = Mathf.Min(effectiveFallY, terrainBottom - terrainFallMargin);
        }
        bool fell = transform.position.y < effectiveFallY;
        if (!tooFar && !fell)
        {
            return;
        }

        Vector3 respawnFrom = transform.position;
        controller.Respawn(spawnPosition, spawnRotation);
        lastAction = "idle";
        lastMode = "wake";
        lastSleepRemaining = 0;
        string reason = fell ? $"fell below {effectiveFallY:F2}" : $"exceeded map radius {maxDistanceFromStart:F1}";
        Debug.Log($"RobotUdpBridge respawned robot: {reason}; from={respawnFrom}; spawn={spawnPosition}.");
    }

    private void RememberCommandTelemetry(RobotCommand command)
    {
        lastFatigue = command.fatigue;
        lastDelusion = command.delusion;
        lastValence = command.valence;
        lastArousal = command.arousal;
        lastTrapPressure = command.trap_pressure;
        lastDopamine = command.dopamine;
        lastDopamineFoodBoost = command.dopamine_food_boost;
        lastMushroomsEaten = command.mushrooms_eaten;
        lastNorepinephrine = command.norepinephrine;
        lastAcetylcholine = command.acetylcholine;
        lastDelusionDrive = command.delusion_drive;
        lastCalciumGate = command.calcium_gate;
        lastHunger = command.hunger;
        lastWorkspaceUnreliable = command.workspace_unreliable;
        lastRealityGateBrakes = command.reality_gate_brakes;
        lastMetaMonitorBrakes = command.meta_monitor_brakes;
        lastHungerAnchorSteps = command.hunger_anchor_steps;
        lastFalseFoodReports = command.false_food_reports;
        lastFalseTrapReports = command.false_trap_reports;
        lastContactProbeSeconds = command.contact_probe_seconds;
        lastSensoryFocus = command.sensory_focus;
        lastSensoryFocusEvents = command.sensory_focus_events;
        lastSurvivalState = string.IsNullOrWhiteSpace(command.survival_state) ? lastSurvivalState : command.survival_state;
        lastCriticalHungerSeconds = command.critical_hunger_seconds;
        lastForageLapseSeconds = command.forage_lapse_seconds;
        lastSurvivalFailures = command.survival_failures;
        lastSurvivalFailureReason = string.IsNullOrWhiteSpace(command.survival_failure_reason) ? lastSurvivalFailureReason : command.survival_failure_reason;
        lastRunSeconds = command.run_seconds;
        lastSecondsSinceFood = command.seconds_since_food;
        lastRouteExploration = command.route_exploration;
        lastIntent = string.IsNullOrWhiteSpace(command.intent) ? lastIntent : command.intent;
        lastGeneration = command.generation <= 0 ? lastGeneration : command.generation;
        lastHandoffEvents = command.handoff_events;
        lastBreakoutEvents = command.breakout_events;
        lastBreakoutPlanRemaining = command.breakout_plan_remaining;
        lastHeadingAction = string.IsNullOrWhiteSpace(command.heading_action) ? lastHeadingAction : command.heading_action;
        lastHeadingTicks = command.heading_ticks;
        lastPhysicsWedgeSeconds = command.physics_wedge_seconds;
        lastUnstuckRespawns = command.unstuck_respawns;
        lastWorkspaceIntent = string.IsNullOrWhiteSpace(command.workspace_intent) ? lastWorkspaceIntent : command.workspace_intent;
        lastWorkspaceProblem = string.IsNullOrWhiteSpace(command.workspace_problem) ? lastWorkspaceProblem : command.workspace_problem;
        lastWorkspaceStrategy = string.IsNullOrWhiteSpace(command.workspace_strategy) ? lastWorkspaceStrategy : command.workspace_strategy;
        lastWorkspaceFeeling = string.IsNullOrWhiteSpace(command.workspace_feeling) ? lastWorkspaceFeeling : command.workspace_feeling;
        lastWorkspaceConfidence = command.workspace_confidence;
        lastWorkspacePromotions = command.workspace_promotions;
        lastShadowEnabled = command.shadow_enabled;
        lastShadowReady = command.shadow_ready;
        lastShadowAction = string.IsNullOrWhiteSpace(command.shadow_action) ? "none" : command.shadow_action;
        lastShadowConfidence = command.shadow_confidence;
        lastShadowEntropy = command.shadow_entropy;
        lastShadowAgreement = command.shadow_agreement;
        lastShadowError = string.IsNullOrWhiteSpace(command.shadow_error) ? "none" : command.shadow_error;
        lastShadowTakeover = command.shadow_takeover;
        lastShadowTakeoverSteps = command.shadow_takeover_steps;
        lastShadowBodySafeActions = command.shadow_body_safe_actions;
        lastShadowMpc = command.shadow_mpc;
        lastShadowMpcEngaged = command.shadow_mpc_engaged;
        lastShadowMpcScore = command.shadow_mpc_score;
        lastShadowMpcMode = string.IsNullOrWhiteSpace(command.shadow_mpc_mode) ? "recurrent" : command.shadow_mpc_mode;
        lastShadowMpcHorizon = command.shadow_mpc_horizon;
        lastShadowMpcDepth = command.shadow_mpc_depth;
        lastShadowMpcUncertaintyStops = command.shadow_mpc_uncertainty_stops;
        lastShadowMpcPlanningFrames = command.shadow_mpc_planning_frames;
        lastShadowMpcCriticalFrames = command.shadow_mpc_critical_frames;
        if (command.food_sensor_radius > 0f)
        {
            lastFoodSensorRadius = command.food_sensor_radius;
        }
        lastTrapCourse = string.IsNullOrWhiteSpace(command.trap_course) ? lastTrapCourse : command.trap_course;
        lastTrapEpisode = command.trap_episode;
        lastTrapSuccesses = command.trap_successes;
        lastTrapFailures = command.trap_failures;
        lastTrapOutcome = string.IsNullOrWhiteSpace(command.trap_outcome) ? lastTrapOutcome : command.trap_outcome;
    }

    private string AvoidBlockedAction(string action)
    {
        string normalized = action.Trim().ToLowerInvariant();
        bool forwardClear = IsClear(transform.forward);
        bool leftClear = IsClear(-transform.right);
        bool rightClear = IsClear(transform.right);

        if ((normalized == "up" || normalized == "forward" || normalized == "move_up") && !forwardClear)
        {
            if (leftClear)
            {
                return "left";
            }

            if (rightClear)
            {
                return "right";
            }

            return "idle";
        }

        if ((normalized == "left" || normalized == "move_left") && !leftClear)
        {
            return rightClear ? "right" : "idle";
        }

        if ((normalized == "right" || normalized == "move_right") && !rightClear)
        {
            return leftClear ? "left" : "idle";
        }

        return action;
    }

    private string VectorToAction(float x, float z)
    {
        Vector2 move = new Vector2(x, z);
        if (move.sqrMagnitude < 0.001f)
        {
            return "idle";
        }

        if (Mathf.Abs(move.x) > Mathf.Abs(move.y))
        {
            return move.x < 0f ? "left" : "right";
        }

        return move.y < 0f ? "down" : "up";
    }

    private void SendState()
    {
        if (sender == null)
        {
            return;
        }

        bool forwardClear = IsClear(transform.forward);
        bool leftClear = IsClear(-transform.right);
        bool rightClear = IsClear(transform.right);
        ObstacleSensor nearestObstacle = SenseForwardObstacle();
        FoodSensor nearestFood = SenseNearestFood();
        lastFoodVisible = nearestFood.visible;
        lastFoodDistance = nearestFood.distance;
        lastObstacleVisible = nearestObstacle.visible;
        lastObstacleDistance = nearestObstacle.distance;
        lastObstacleOffsetX = nearestObstacle.offsetX;
        if (nearestFood.visible && !wasFoodVisible)
        {
            foodSightings += 1;
        }
        wasFoodVisible = nearestFood.visible;
        Vector2 move = controller.LastMoveInput;
        Vector3 position = transform.position;
        RobotState state = new RobotState
        {
            control_mode = controller.ControlMode.ToString(),
            action = lastAction,
            mode = lastMode,
            animation = controller.GetCurrentAnimatorStateName(),
            sleeping = controller.IsSleeping,
            grounded = controller.IsGrounded,
            forward_clear = forwardClear,
            left_clear = leftClear,
            right_clear = rightClear,
            blocked = !forwardClear,
            horizontal_collision = (controller.LastCollisionFlags & CollisionFlags.Sides) != 0,
            obstacle_visible = nearestObstacle.visible,
            obstacle_distance = nearestObstacle.distance,
            obstacle_offset_x = nearestObstacle.offsetX,
            avoid_move_x = nearestObstacle.avoidInput.x,
            avoid_move_z = nearestObstacle.avoidInput.y,
            fatigue = lastFatigue,
            dopamine = controlDopamine,
            norepinephrine = controlNorepinephrine,
            acetylcholine = controlAcetylcholine,
            delusion_drive = controlNoiseInjection,
            calcium_gate = controlCalciumGate,
            route_exploration = controlRouteExploration,
            ate_mushroom = pendingMushroomPickups,
            mushroom_reward = pendingMushroomReward,
            mushroom_pickups_total = totalMushroomPickups,
            mushroom_reward_total = totalMushroomReward,
            food_visible = nearestFood.visible,
            food_distance = nearestFood.distance,
            food_dir_x = nearestFood.direction.x,
            food_dir_z = nearestFood.direction.z,
            food_move_x = nearestFood.moveInput.x,
            food_move_z = nearestFood.moveInput.y,
            food_world_x = nearestFood.worldDirection.x,
            food_world_z = nearestFood.worldDirection.z,
            food_available_in_radius = lastAvailableFoodWithinRadius,
            food_visible_in_radius = lastVisibleFoodWithinRadius,
            food_occluded_in_radius = lastOccludedFoodWithinRadius,
            nearest_available_food_distance = lastNearestAvailableFoodDistance,
            directional_rays = SenseDirectionalRays(),
            directional_body_clearance = SenseDirectionalBodyClearance(),
            trap_course = TrapCourseSpawner.CurrentCourseLabel,
            trap_episode = TrapCourseSpawner.CurrentEpisode,
            trap_successes = TrapCourseSpawner.CourseSuccesses,
            trap_failures = TrapCourseSpawner.CourseFailures,
            trap_outcome = TrapCourseSpawner.CurrentOutcome,
            sleep_remaining = lastSleepRemaining,
            x = position.x,
            y = position.y,
            z = position.z,
            yaw = transform.eulerAngles.y,
            move_x = move.x,
            move_z = move.y,
        };

        string json = JsonUtility.ToJson(state);
        byte[] bytes = Encoding.UTF8.GetBytes(json);
        sender.Send(bytes, bytes.Length, sendEndPoint);
        lastMushroomPickups = pendingMushroomPickups;
        lastMushroomReward = pendingMushroomReward;
        pendingMushroomPickups = 0;
        pendingMushroomReward = 0f;
    }

    public void RegisterMushroomPickup(float dopamineReward)
    {
        pendingMushroomPickups += 1;
        pendingMushroomReward = Mathf.Clamp01(pendingMushroomReward + dopamineReward);
        totalMushroomPickups += 1;
        totalMushroomReward += Mathf.Max(0f, dopamineReward);
    }

    public void SetSpawnPoint(Vector3 position, Quaternion rotation)
    {
        spawnPosition = position;
        spawnRotation = rotation;
    }

    public void RestoreTerrainSpawnPoint(bool respawnRobot)
    {
        Vector3 position = terrainSpawnPosition;
        Terrain terrain = Terrain.activeTerrain;
        if (terrain != null)
        {
            position.y = terrain.SampleHeight(position) + terrain.transform.position.y + 0.25f;
        }
        spawnPosition = position;
        spawnRotation = terrainSpawnRotation;
        if (respawnRobot)
        {
            hasAiMoveTarget = false;
            controller.Respawn(spawnPosition, spawnRotation);
            lastAction = "idle";
            lastMode = "wake";
            lastSleepRemaining = 0;
        }
    }

    private struct FoodSensor
    {
        public bool visible;
        public float distance;
        public Vector3 direction;
        public Vector2 moveInput;
        public Vector3 worldDirection;
    }

    private struct ObstacleSensor
    {
        public bool visible;
        public float distance;
        public float offsetX;
        public Vector2 avoidInput;
    }

    private ObstacleSensor SenseForwardObstacle()
    {
        ObstacleSensor result = new ObstacleSensor { visible = false, distance = 0f, offsetX = 0f, avoidInput = Vector2.zero };
        Vector3 origin = transform.position + Vector3.up * sensorHeight;
        float[] angles = { -35f, -18f, 0f, 18f, 35f };
        float bestDistance = obstacleAwarenessDistance;
        float bestOffset = 0f;

        foreach (float angle in angles)
        {
            Vector3 direction = Quaternion.AngleAxis(angle, Vector3.up) * transform.forward;
            if (!Physics.Raycast(origin, direction, out RaycastHit hit, obstacleAwarenessDistance, obstacleMask, QueryTriggerInteraction.Ignore))
            {
                continue;
            }

            if (hit.transform == transform || hit.transform.IsChildOf(transform))
            {
                continue;
            }

            if (hit.distance < bestDistance)
            {
                bestDistance = hit.distance;
                bestOffset = Mathf.Clamp(angle / 35f, -1f, 1f);
            }
        }

        if (bestDistance < obstacleAwarenessDistance)
        {
            float avoidSide = bestOffset >= 0f ? -1f : 1f;
            Vector3 avoidWorld = (transform.forward * 0.80f + transform.right * avoidSide * 0.62f).normalized;
            result.visible = true;
            result.distance = bestDistance;
            result.offsetX = bestOffset;
            result.avoidInput = WorldDirectionToCameraInput(avoidWorld);
        }

        return result;
    }

    private FoodSensor SenseNearestFood()
    {
        FoodSensor result = new FoodSensor { visible = false, distance = 0f, direction = Vector3.zero, moveInput = Vector2.zero, worldDirection = Vector3.zero };
        FoodMushroom[] foods = FindObjectsByType<FoodMushroom>(FindObjectsInactive.Exclude);
        float sensingRadius = TrapCourseSpawner.IsActive
            ? Mathf.Clamp(lastFoodSensorRadius, 7f, 13f)
            : Mathf.Clamp(lastFoodSensorRadius, foodSensorRadius, 28f);
        float bestDistance = sensingRadius;
        Vector3 bestDirection = Vector3.zero;
        Vector3 origin = transform.position;

        lastAvailableFoodWithinRadius = 0;
        lastVisibleFoodWithinRadius = 0;
        lastOccludedFoodWithinRadius = 0;
        lastNearestAvailableFoodDistance = -1f;
        foreach (FoodMushroom food in foods)
        {
            if (!TrapCourseSpawner.IsRelevantFood(food) || !food.isActiveAndEnabled || !food.IsAvailable)
            {
                continue;
            }
            Vector3 diagnosticOffset = food.transform.position - origin;
            diagnosticOffset.y = 0f;
            float diagnosticDistance = diagnosticOffset.magnitude;
            if (diagnosticDistance <= 0.001f || diagnosticDistance > sensingRadius)
            {
                continue;
            }
            lastAvailableFoodWithinRadius += 1;
            if (lastNearestAvailableFoodDistance < 0f || diagnosticDistance < lastNearestAvailableFoodDistance)
            {
                lastNearestAvailableFoodDistance = diagnosticDistance;
            }
            if (HasFoodLineOfSight(food, diagnosticDistance))
            {
                lastVisibleFoodWithinRadius += 1;
            }
            else
            {
                lastOccludedFoodWithinRadius += 1;
            }
        }

        if (lockedFoodTarget != null && TrapCourseSpawner.IsRelevantFood(lockedFoodTarget) && lockedFoodTarget.isActiveAndEnabled && lockedFoodTarget.IsAvailable)
        {
            Vector3 lockedOffset = lockedFoodTarget.transform.position - origin;
            lockedOffset.y = 0f;
            float lockedDistance = lockedOffset.magnitude;
            if (lockedDistance > 0.001f && lockedDistance <= sensingRadius + 2f && HasFoodLineOfSight(lockedFoodTarget, lockedDistance))
            {
                return BuildFoodSensor(lockedOffset.normalized, lockedDistance);
            }
        }

        lockedFoodTarget = null;
        foreach (FoodMushroom food in foods)
        {
            if (!TrapCourseSpawner.IsRelevantFood(food) || !food.isActiveAndEnabled || !food.IsAvailable)
            {
                continue;
            }

            Vector3 offset = food.transform.position - origin;
            offset.y = 0f;
            float distance = offset.magnitude;
            if (distance <= 0.001f || distance > bestDistance)
            {
                continue;
            }
            if (!HasFoodLineOfSight(food, distance))
            {
                continue;
            }

            bestDistance = distance;
            bestDirection = offset.normalized;
            lockedFoodTarget = food;
        }

        if (bestDistance < sensingRadius)
        {
            result = BuildFoodSensor(bestDirection, bestDistance);
        }

        return result;
    }

    private bool HasFoodLineOfSight(FoodMushroom food, float horizontalDistance)
    {
        Vector3 origin = transform.position + Vector3.up * sensorHeight;
        Vector3 target = food.transform.position + Vector3.up * sensorHeight;
        Vector3 offset = target - origin;
        float distance = offset.magnitude;
        if (distance <= 0.001f)
        {
            return true;
        }
        RaycastHit[] hits = Physics.RaycastAll(origin, offset / distance, distance, obstacleMask, QueryTriggerInteraction.Ignore);
        foreach (RaycastHit hit in hits)
        {
            if (hit.transform == transform || hit.transform.IsChildOf(transform) || hit.transform == food.transform || hit.transform.IsChildOf(food.transform))
            {
                continue;
            }
            if (hit.distance < distance - 0.1f)
            {
                return false;
            }
        }
        return horizontalDistance <= 7f || !TrapCourseSpawner.IsActive;
    }

    private FoodSensor BuildFoodSensor(Vector3 worldDirection, float distance)
    {
        FoodSensor result = new FoodSensor
        {
            visible = true,
            distance = distance,
            direction = transform.InverseTransformDirection(worldDirection),
            moveInput = WorldDirectionToCameraInput(worldDirection),
            worldDirection = worldDirection
        };
        result.direction.y = 0f;
        return result;
    }

    private Vector2 WorldDirectionToCameraInput(Vector3 direction)
    {
        Vector3 forward = Vector3.forward;
        Vector3 right = Vector3.right;
        if (Camera.main != null)
        {
            forward = Camera.main.transform.forward;
            right = Camera.main.transform.right;
            forward.y = 0f;
            right.y = 0f;
            forward.Normalize();
            right.Normalize();
        }

        Vector2 input = new Vector2(Vector3.Dot(direction, right), Vector3.Dot(direction, forward));
        return Vector2.ClampMagnitude(input, 1f);
    }

    public string GetHudText()
    {
        bool bodyContact = (controller.LastCollisionFlags & CollisionFlags.Sides) != 0;
        string forwardSensor = !IsClear(transform.forward) ? "blocked" : "clear";
        string leftSensor = !IsClear(-transform.right) ? "blocked" : "clear";
        string rightSensor = !IsClear(transform.right) ? "blocked" : "clear";
        string bodySensor = bodyContact ? "contact" : "clear";
        string shadowState = !lastShadowEnabled
            ? "OFF"
            : lastShadowReady
                ? $"{lastShadowAction} ({(lastShadowAgreement >= 0.85f ? "AGREE" : "DIVERGE")})"
                : $"WAITING: {lastShadowError}";
        return
            $"Intent: {lastIntent}\n" +
            $"Motor Control: {(lastShadowTakeover ? (lastAction == "gru_course" ? "GRU COURSE" : lastAction == "gru_terrain" ? "GRU TERRAIN" : "GRU FOOD") : "STABLE CONTROLLER")}  action {lastAction}\n" +
            $"GRU Shadow ({(lastShadowTakeover ? "ACTIVE" : "PASSIVE")}): {shadowState}  confidence {lastShadowConfidence:0.00}\n" +
            $"MPC: {(lastShadowMpc ? lastShadowMpcMode.ToUpperInvariant() : "OFF")}  H{lastShadowMpcHorizon} D{lastShadowMpcDepth:0.0} U{lastShadowMpcUncertaintyStops}  score {lastShadowMpcScore:+0.00;-0.00;0.00}\n" +
            $"Shadow Agreement: {lastShadowAgreement:0.00}  entropy {lastShadowEntropy:0.00}\n" +
            $"Body-Clear Actions: {lastShadowBodySafeActions}/8\n" +
            $"Learned Control Frames: {lastShadowTakeoverSteps}\n" +
            $"Course: {lastTrapCourse} #{lastTrapEpisode}  {lastTrapOutcome}  Wins {lastTrapSuccesses} / Timeouts {lastTrapFailures}\n" +
            $"Workspace: {lastWorkspaceProblem} / {lastWorkspaceStrategy} ({lastWorkspaceConfidence:0.00})\n" +
            $"Workspace Feeling: {lastWorkspaceFeeling}  Promotions: {lastWorkspacePromotions}\n" +
            $"Action: {lastAction}  Mode: {lastMode}\n" +
            $"Valence: {lastValence:+0.00;-0.00;0.00}  Arousal: {lastArousal:0.00}\n" +
            $"Fatigue: {lastFatigue:0.00}  Delusion: {lastDelusion:0.00}\n" +
            $"Dopamine: {lastDopamine:0.00}  Food Boost: {lastDopamineFoodBoost:0.00}  Food: {lastMushroomsEaten} eaten / {foodSightings} sensor locks\n" +
            $"Hunger: {lastHunger:0.00}  Food Radius: {lastFoodSensorRadius:0}  Calcium: {lastCalciumGate:0.00}\n" +
            $"Survival: {lastSurvivalState}  run {lastRunSeconds:0}s  since food {lastSecondsSinceFood:0}s\n" +
            $"Stress: critical {lastCriticalHungerSeconds:0}s  lapse {lastForageLapseSeconds:0}s\n" +
            $"Failures: {lastSurvivalFailures}  Reason: {lastSurvivalFailureReason}\n" +
            $"Grounding Events: reality {lastRealityGateBrakes}  meta {lastMetaMonitorBrakes}  hunger-anchor {lastHungerAnchorSteps}\n" +
            $"Sensory Focus: {lastSensoryFocus:0.00}  events {lastSensoryFocusEvents}\n" +
            $"Contact Probe: {lastContactProbeSeconds:0.0}s\n" +
            $"Physics Wedge: {lastPhysicsWedgeSeconds:0.0}s  Respawns: {lastUnstuckRespawns}\n" +
            $"False Reports: food {lastFalseFoodReports}  trap {lastFalseTrapReports}\n" +
            $"Mushroom Reward: last {lastMushroomReward:0.00}  total {totalMushroomReward:0.00}\n" +
            $"Food Sensor: {(lastFoodVisible ? "visible" : "none")}  Distance: {lastFoodDistance:0.0}\n" +
            $"Obstacle Sensor: {(lastObstacleVisible ? "visible" : "none")}  Distance: {lastObstacleDistance:0.0}  Offset: {lastObstacleOffsetX:+0.00;-0.00;0.00}\n" +
            $"Heading: {lastHeadingAction} ({lastHeadingTicks})  Trap: {lastTrapPressure:0.00}  Breakouts: {lastBreakoutEvents}  Plan: {lastBreakoutPlanRemaining}\n" +
            $"Generation: {lastGeneration}  Handoffs: {lastHandoffEvents}\n" +
            $"Sensors: L {leftSensor}  F {forwardSensor}  R {rightSensor}  Body {bodySensor}";
    }

    public void SetHudControls(float dopamine, float norepinephrine, float acetylcholine, float noiseInjection, float calciumGate, float routeExploration)
    {
        controlDopamine = Mathf.Clamp01(dopamine);
        controlNorepinephrine = Mathf.Clamp01(norepinephrine);
        controlAcetylcholine = Mathf.Clamp01(acetylcholine);
        controlNoiseInjection = Mathf.Clamp01(noiseInjection);
        controlCalciumGate = Mathf.Clamp01(calciumGate);
        controlRouteExploration = Mathf.Clamp01(routeExploration);
    }

    private bool IsClear(Vector3 direction)
    {
        Vector3 origin = transform.position + Vector3.up * sensorHeight;
        Vector3 flatDirection = direction;
        flatDirection.y = 0f;
        if (flatDirection.sqrMagnitude < 0.001f)
        {
            return true;
        }

        flatDirection.Normalize();
        return !Physics.Raycast(origin, flatDirection, obstacleSensorDistance, obstacleMask, QueryTriggerInteraction.Ignore);
    }

    private float[] SenseDirectionalRays()
    {
        Vector3[] directions =
        {
            Vector3.forward,
            (Vector3.forward + Vector3.right).normalized,
            Vector3.right,
            (Vector3.back + Vector3.right).normalized,
            Vector3.back,
            (Vector3.back + Vector3.left).normalized,
            Vector3.left,
            (Vector3.forward + Vector3.left).normalized,
        };
        float[] distances = new float[directions.Length];
        Vector3 origin = transform.position + Vector3.up * sensorHeight;
        float range = Mathf.Max(0.1f, shadowRayDistance);
        for (int index = 0; index < directions.Length; index++)
        {
            distances[index] = 1f;
            RaycastHit[] hits = Physics.RaycastAll(origin, directions[index], range, obstacleMask, QueryTriggerInteraction.Ignore);
            float nearest = range;
            foreach (RaycastHit hit in hits)
            {
                if (hit.transform == transform || hit.transform.IsChildOf(transform))
                {
                    continue;
                }
                nearest = Mathf.Min(nearest, hit.distance);
            }
            if (TrapCourseSpawner.IsActive)
            {
                const float edgeStep = 0.5f;
                for (float distance = edgeStep; distance <= range; distance += edgeStep)
                {
                    Vector3 groundOrigin = transform.position + directions[index] * distance + Vector3.up * 0.75f;
                    if (!HasExternalGroundBelow(groundOrigin))
                    {
                        nearest = Mathf.Min(nearest, Mathf.Max(0f, distance - edgeStep));
                        break;
                    }
                }
            }
            distances[index] = Mathf.Clamp01(nearest / range);
        }
        return distances;
    }

    private float[] SenseDirectionalBodyClearance()
    {
        Vector3[] directions =
        {
            Vector3.forward,
            (Vector3.forward + Vector3.right).normalized,
            Vector3.right,
            (Vector3.back + Vector3.right).normalized,
            Vector3.back,
            (Vector3.back + Vector3.left).normalized,
            Vector3.left,
            (Vector3.forward + Vector3.left).normalized,
        };
        float[] clearance = new float[directions.Length];
        if (bodyCollider == null)
        {
            for (int index = 0; index < clearance.Length; index++)
            {
                clearance[index] = 1f;
            }
            return clearance;
        }

        Vector3 scale = transform.lossyScale;
        float radiusScale = Mathf.Max(Mathf.Abs(scale.x), Mathf.Abs(scale.z));
        float radius = Mathf.Max(0.02f, bodyCollider.radius * radiusScale - bodyCollider.skinWidth);
        float height = Mathf.Max(radius * 2f, bodyCollider.height * Mathf.Abs(scale.y));
        Vector3 center = transform.TransformPoint(bodyCollider.center);
        float halfSegment = Mathf.Max(0f, height * 0.5f - radius);
        Vector3 bottom = center - Vector3.up * halfSegment;
        Vector3 top = center + Vector3.up * halfSegment;
        float distance = Mathf.Max(0.05f, shadowBodyProbeStep);

        for (int index = 0; index < directions.Length; index++)
        {
            clearance[index] = 1f;
            RaycastHit[] hits = Physics.CapsuleCastAll(
                bottom,
                top,
                radius,
                directions[index],
                distance,
                obstacleMask,
                QueryTriggerInteraction.Ignore
            );
            foreach (RaycastHit hit in hits)
            {
                if (hit.transform == transform || hit.transform.IsChildOf(transform))
                {
                    continue;
                }
                if (hit.normal.y > 0.55f)
                {
                    continue;
                }
                clearance[index] = 0f;
                break;
            }
        }
        return clearance;
    }

    private bool HasExternalGroundBelow(Vector3 origin)
    {
        RaycastHit[] hits = Physics.RaycastAll(
            origin,
            Vector3.down,
            shadowGroundProbeDepth,
            obstacleMask,
            QueryTriggerInteraction.Ignore
        );
        foreach (RaycastHit hit in hits)
        {
            if (hit.transform == transform || hit.transform.IsChildOf(transform))
            {
                continue;
            }
            if (Vector3.Dot(hit.normal, Vector3.up) > 0.35f)
            {
                return true;
            }
        }
        return false;
    }

    private void UpdateAiSteering()
    {
        if (!hasAiMoveTarget || controller.IsBodyLocked)
        {
            return;
        }
        float smoothTime = Mathf.Max(0.01f, aiSteeringBlendSeconds);
        smoothedAiMove = Vector2.SmoothDamp(
            smoothedAiMove,
            desiredAiMove,
            ref aiMoveVelocity,
            smoothTime,
            Mathf.Infinity,
            Time.deltaTime
        );
        controller.SetAiMove(smoothedAiMove, desiredAiRun);
    }

    private void CloseSockets()
    {
        receiver?.Close();
        receiver = null;
        sender?.Close();
        sender = null;
    }
}
