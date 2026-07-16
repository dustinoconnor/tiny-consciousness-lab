using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;

public class TrapCourseSpawner : MonoBehaviour
{
    public enum CourseType
    {
        UTrap,
        CTrap,
        LWall,
        Zigzag,
        OffsetBarriers,
        NarrowCorridor,
    }

    public static string CurrentCourseLabel { get; private set; } = "natural_terrain";
    public static int CurrentEpisode { get; private set; }
    public static int CourseSuccesses { get; private set; }
    public static int CourseFailures { get; private set; }
    public static string CurrentOutcome { get; private set; } = "inactive";
    public static bool IsActive => activeSpawner != null;

    private static TrapCourseSpawner activeSpawner;

    [SerializeField] private ThirdPersonRobotController robot;
    [SerializeField] private FoodMushroom foodPrefab;
    [SerializeField] private CameraLookAtCharacter courseCamera;
    [SerializeField] private CourseType course = CourseType.UTrap;
    [SerializeField] private bool buildOnStart = true;
    [SerializeField] private bool snapDeckToTerrain = true;
    [SerializeField] private bool advanceAfterGoal = true;
    [SerializeField] private float advanceDelaySeconds = 2f;
    [SerializeField] private float episodeTimeoutSeconds = 90f;
    [SerializeField] private Vector2 deckSize = new Vector2(34f, 38f);
    [SerializeField] private float deckThickness = 0.5f;
    [SerializeField] private float wallHeight = 2.8f;
    [SerializeField] private float wallThickness = 0.75f;
    [SerializeField] private Material deckMaterial;
    [SerializeField] private Material wallMaterial;

    private readonly List<GameObject> generated = new List<GameObject>();
    private Transform generatedRoot;
    private Vector3 robotStart;
    private Vector3 foodPosition;
    private float advanceAt = -1f;
    private FoodMushroom goalFood;
    private float episodeStartedAt;

    public static bool IsRelevantFood(FoodMushroom food)
    {
        return activeSpawner == null || food == activeSpawner.goalFood;
    }

    private void Start()
    {
        if (buildOnStart)
        {
            BuildCourse();
        }
    }

    private void OnDisable()
    {
        if (!Application.isPlaying)
        {
            return;
        }
        ClearCourse();
        if (robot == null)
        {
            robot = FindAnyObjectByType<ThirdPersonRobotController>();
        }
        RobotUdpBridge bridge = robot != null ? robot.GetComponent<RobotUdpBridge>() : null;
        if (bridge != null)
        {
            bridge.RestoreTerrainSpawnPoint(true);
        }
    }

    private void Update()
    {
        if (advanceAt > 0f && Time.time >= advanceAt)
        {
            advanceAt = -1f;
            NextCourse();
            return;
        }
        if (generatedRoot != null && CurrentOutcome == "running" && Time.time - episodeStartedAt >= episodeTimeoutSeconds)
        {
            CourseFailures += 1;
            CurrentOutcome = "timeout";
            advanceAt = Time.time + Mathf.Max(0.25f, advanceDelaySeconds);
        }
        Keyboard keyboard = Keyboard.current;
        if (keyboard == null)
        {
            return;
        }
        if (keyboard.nKey.wasPressedThisFrame)
        {
            NextCourse();
        }
        else if (keyboard.bKey.wasPressedThisFrame)
        {
            PreviousCourse();
        }
        else if (keyboard.rKey.wasPressedThisFrame)
        {
            ResetRobot();
        }
    }

    [ContextMenu("Build Course")]
    public void BuildCourse()
    {
        ClearCourse();
        float deckY = transform.position.y;
        if (snapDeckToTerrain && Terrain.activeTerrain != null)
        {
            deckY = Terrain.activeTerrain.SampleHeight(transform.position) + Terrain.activeTerrain.transform.position.y + 0.35f;
        }

        generatedRoot = new GameObject($"Generated Trap Course - {course}").transform;
        activeSpawner = this;
        generatedRoot.SetParent(transform, false);
        generatedRoot.position = new Vector3(transform.position.x, deckY, transform.position.z);
        CreateBlock("Test Deck", new Vector3(0f, -deckThickness * 0.5f, 0f), new Vector3(deckSize.x, deckThickness, deckSize.y), deckMaterial, true);
        BuildPerimeter();

        switch (course)
        {
            case CourseType.UTrap: BuildUTrap(); break;
            case CourseType.CTrap: BuildCTrap(); break;
            case CourseType.LWall: BuildLWall(); break;
            case CourseType.Zigzag: BuildZigzag(); break;
            case CourseType.OffsetBarriers: BuildOffsetBarriers(); break;
            case CourseType.NarrowCorridor: BuildNarrowCorridor(); break;
        }

        SpawnFood();
        CurrentCourseLabel = course.ToString().ToLowerInvariant();
        CurrentEpisode += 1;
        CurrentOutcome = "running";
        episodeStartedAt = Time.time;
        ResetRobot();
    }

    [ContextMenu("Next Course")]
    public void NextCourse()
    {
        CourseType[] values = (CourseType[])Enum.GetValues(typeof(CourseType));
        course = values[(Array.IndexOf(values, course) + 1) % values.Length];
        BuildCourse();
    }

    [ContextMenu("Previous Course")]
    public void PreviousCourse()
    {
        CourseType[] values = (CourseType[])Enum.GetValues(typeof(CourseType));
        int index = Array.IndexOf(values, course) - 1;
        course = values[index < 0 ? values.Length - 1 : index];
        BuildCourse();
    }

    [ContextMenu("Reset Robot")]
    public void ResetRobot()
    {
        if (robot == null)
        {
            robot = FindAnyObjectByType<ThirdPersonRobotController>();
        }
        if (robot == null || generatedRoot == null)
        {
            return;
        }
        Vector3 worldStart = generatedRoot.TransformPoint(robotStart);
        Quaternion worldRotation = Quaternion.LookRotation(generatedRoot.forward, Vector3.up);
        RobotUdpBridge bridge = robot.GetComponent<RobotUdpBridge>();
        if (bridge != null)
        {
            bridge.SetSpawnPoint(worldStart, worldRotation);
        }
        robot.Respawn(worldStart, worldRotation);
        if (courseCamera == null)
        {
            courseCamera = FindAnyObjectByType<CameraLookAtCharacter>();
        }
        if (courseCamera != null)
        {
            courseCamera.SetTargetAndSnap(robot.transform);
        }
    }

    public void RegisterGoalPickup(FoodMushroom food)
    {
        if (!advanceAfterGoal || advanceAt > 0f || food == null || food.transform.parent != generatedRoot)
        {
            return;
        }
        CourseSuccesses += 1;
        CurrentOutcome = "success";
        advanceAt = Time.time + Mathf.Max(0.25f, advanceDelaySeconds);
    }

    [ContextMenu("Clear Course")]
    public void ClearCourse()
    {
        foreach (GameObject item in generated)
        {
            if (item != null)
            {
                Destroy(item);
            }
        }
        generated.Clear();
        if (generatedRoot != null)
        {
            Destroy(generatedRoot.gameObject);
        }
        generatedRoot = null;
        goalFood = null;
        if (activeSpawner == this)
        {
            activeSpawner = null;
        }
        CurrentCourseLabel = "natural_terrain";
        CurrentOutcome = "inactive";
    }

    private void BuildUTrap()
    {
        Wall(0f, 3f, 6f, wallThickness);
        Wall(-2.65f, 0f, wallThickness, 6f);
        Wall(2.65f, 0f, wallThickness, 6f);
        robotStart = new Vector3(0f, 0.25f, -1.9f);
        foodPosition = new Vector3(0f, 0.4f, 5.5f);
    }

    private void BuildPerimeter()
    {
        float edgeHeight = 1.5f;
        float edgeThickness = 0.65f;
        CreateBlock("Deck Boundary", new Vector3(0f, edgeHeight * 0.5f, deckSize.y * 0.5f), new Vector3(deckSize.x, edgeHeight, edgeThickness), wallMaterial, false);
        CreateBlock("Deck Boundary", new Vector3(0f, edgeHeight * 0.5f, -deckSize.y * 0.5f), new Vector3(deckSize.x, edgeHeight, edgeThickness), wallMaterial, false);
        CreateBlock("Deck Boundary", new Vector3(deckSize.x * 0.5f, edgeHeight * 0.5f, 0f), new Vector3(edgeThickness, edgeHeight, deckSize.y), wallMaterial, false);
        CreateBlock("Deck Boundary", new Vector3(-deckSize.x * 0.5f, edgeHeight * 0.5f, 0f), new Vector3(edgeThickness, edgeHeight, deckSize.y), wallMaterial, false);
    }

    private void BuildCTrap()
    {
        Wall(-4.5f, 0f, wallThickness, 11f);
        Wall(0f, 5.1f, 9f, wallThickness);
        Wall(0f, -5.1f, 9f, wallThickness);
        robotStart = new Vector3(-1.5f, 0.25f, 0f);
        foodPosition = new Vector3(-8f, 0.4f, 0f);
    }

    private void BuildLWall()
    {
        Wall(0f, 3f, 10f, wallThickness);
        Wall(4.6f, -1.5f, wallThickness, 9f);
        robotStart = new Vector3(1.8f, 0.25f, 0f);
        foodPosition = new Vector3(1.8f, 0.4f, 7.5f);
    }

    private void BuildZigzag()
    {
        Wall(-2.7f, -4f, wallThickness, 11f);
        Wall(2.7f, 3f, wallThickness, 11f);
        Wall(-2.7f, 10f, wallThickness, 8f);
        robotStart = new Vector3(0f, 0.25f, -10f);
        foodPosition = new Vector3(0f, 0.4f, 14f);
    }

    private void BuildOffsetBarriers()
    {
        Wall(-3.3f, -3f, 9f, wallThickness);
        Wall(3.3f, 3f, 9f, wallThickness);
        Wall(-3.3f, 9f, 9f, wallThickness);
        robotStart = new Vector3(0f, 0.25f, -10f);
        foodPosition = new Vector3(0f, 0.4f, 14f);
    }

    private void BuildNarrowCorridor()
    {
        Wall(-2f, -4f, wallThickness, 14f);
        Wall(2f, -1f, wallThickness, 8f);
        Wall(5.5f, 3f, 7.5f, wallThickness);
        Wall(8.9f, 7f, wallThickness, 8f);
        robotStart = new Vector3(0f, 0.25f, -10f);
        foodPosition = new Vector3(7f, 0.4f, 11f);
    }

    private void Wall(float x, float z, float width, float depth)
    {
        CreateBlock("Course Wall", new Vector3(x, wallHeight * 0.5f, z), new Vector3(width, wallHeight, depth), wallMaterial, false);
    }

    private GameObject CreateBlock(string label, Vector3 localPosition, Vector3 scale, Material material, bool deck)
    {
        GameObject block = GameObject.CreatePrimitive(PrimitiveType.Cube);
        block.name = label;
        block.transform.SetParent(generatedRoot, false);
        block.transform.localPosition = localPosition;
        block.transform.localScale = scale;
        Renderer renderer = block.GetComponent<Renderer>();
        if (material != null)
        {
            renderer.sharedMaterial = material;
        }
        else
        {
            renderer.material.color = deck ? new Color(0.24f, 0.28f, 0.29f) : new Color(0.38f, 0.42f, 0.40f);
        }
        generated.Add(block);
        return block;
    }

    private void SpawnFood()
    {
        FoodMushroom source = foodPrefab != null ? foodPrefab : FindAnyObjectByType<FoodMushroom>();
        if (source == null)
        {
            Debug.LogWarning("TrapCourseSpawner needs a FoodMushroom prefab or scene instance for its goal.");
            return;
        }
        FoodMushroom food = Instantiate(source, generatedRoot);
        food.name = "Trap Course Goal Mushroom";
        food.transform.localPosition = foodPosition;
        food.gameObject.SetActive(true);
        goalFood = food;
        generated.Add(food.gameObject);
    }
}
