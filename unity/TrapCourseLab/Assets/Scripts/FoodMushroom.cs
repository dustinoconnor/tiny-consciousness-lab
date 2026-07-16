using UnityEngine;

[RequireComponent(typeof(BoxCollider))]
[RequireComponent(typeof(Rigidbody))]
public class FoodMushroom : MonoBehaviour
{
    private const float DefaultDopamineReward = 0.35f;

    [SerializeField, Range(0f, 1f)] private float dopamineReward = 0.35f;
    [SerializeField] private float respawnSeconds = 45f;

    private Collider pickupCollider;
    private Rigidbody body;
    private Renderer[] renderers;
    private float respawnAt = -1f;
    private bool available = true;

    public bool IsAvailable => available;

    private void Awake()
    {
        pickupCollider = GetComponent<Collider>();
        pickupCollider.isTrigger = true;

        body = GetComponent<Rigidbody>();
        body.isKinematic = true;
        body.useGravity = false;

        renderers = GetComponentsInChildren<Renderer>();
    }

    private void Update()
    {
        if (respawnAt > 0f && Time.time >= respawnAt)
        {
            SetAvailable(true);
            respawnAt = -1f;
        }
    }

    private void OnTriggerEnter(Collider other)
    {
        RobotUdpBridge bridge = other.GetComponentInParent<RobotUdpBridge>();
        if (bridge == null)
        {
            return;
        }

        bridge.RegisterMushroomPickup(dopamineReward > 0f ? dopamineReward : DefaultDopamineReward);
        TrapCourseSpawner course = GetComponentInParent<TrapCourseSpawner>();
        if (course != null)
        {
            course.RegisterGoalPickup(this);
        }
        if (respawnSeconds > 0f)
        {
            SetAvailable(false);
            respawnAt = Time.time + respawnSeconds;
        }
        else
        {
            gameObject.SetActive(false);
        }
    }

    private void SetAvailable(bool available)
    {
        this.available = available;
        pickupCollider.enabled = available;
        foreach (Renderer item in renderers)
        {
            item.enabled = available;
        }
    }
}
