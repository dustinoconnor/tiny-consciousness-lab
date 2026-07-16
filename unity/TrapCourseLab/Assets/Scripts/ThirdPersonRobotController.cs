using UnityEngine;
using UnityEngine.InputSystem;

[RequireComponent(typeof(CharacterController))]
public class ThirdPersonRobotController : MonoBehaviour
{
    public enum RobotControlMode
    {
        Manual,
        Auto
    }

    [SerializeField] private Animator animator;
    [SerializeField] private Transform cameraTransform;
    [SerializeField] private RobotControlMode controlMode = RobotControlMode.Manual;
    [SerializeField] private float walkSpeed = 2.2f;
    [SerializeField] private float runSpeed = 4.5f;
    [SerializeField] private float rotationSharpness = 14f;
    [SerializeField] private float jumpHeight = 1.25f;
    [SerializeField] private float gravity = -20f;
    [SerializeField] private float groundedStickVelocity = -2f;
    [SerializeField] private float groundedActionGraceTime = 0.12f;
    [SerializeField] private AttackVfxController attackVfx;
    [SerializeField] private bool syncLandingToJumpClip = true;
    [SerializeField, Range(0f, 1f)] private float landingClipNormalizedTime = 0.6f;
    [SerializeField, Range(0f, 1f)] private float flipStartNormalizedTime = 0.28f;
    [SerializeField, Range(0f, 1f)] private float flipExitNormalizedTime = 0.78f;
    [SerializeField, Range(0f, 1f)] private float airAnimationExitTime = 0.88f;
    [SerializeField, Range(0f, 1f)] private float punchStartNormalizedTime = 0.15f;
    [SerializeField, Range(0f, 1f)] private float kickStartNormalizedTime = 0.12f;
    [SerializeField, Range(0f, 1f)] private float punchExitNormalizedTime = 0.75f;
    [SerializeField, Range(0f, 1f)] private float kickExitNormalizedTime = 0.65f;
    [SerializeField] private float attackRestartLockTime = 0.18f;
    [SerializeField] private float punchRepeatDelay = 0.65f;
    [SerializeField] private float kickRepeatDelay = 0.7f;
    [SerializeField] private float getUpMovementLockSeconds = 2.0f;
    [SerializeField] private float poseRecoveryLockSeconds = 0.75f;

    private static readonly int SpeedHash = Animator.StringToHash("Speed");
    private static readonly int JumpHash = Animator.StringToHash("Jump");
    private static readonly int FlipHash = Animator.StringToHash("Flip");
    private static readonly int PunchHash = Animator.StringToHash("Punch");
    private static readonly int KickHash = Animator.StringToHash("Kick");
    private static readonly int SleepHash = Animator.StringToHash("Sleep");
    private static readonly int WakeHash = Animator.StringToHash("Wake");
    private static readonly int GroundedHash = Animator.StringToHash("Grounded");
    private static readonly int VerticalSpeedHash = Animator.StringToHash("VerticalSpeed");
    private static readonly int IdleStateHash = Animator.StringToHash("Idle");
    private static readonly int WalkStateHash = Animator.StringToHash("Walk");
    private static readonly int RunStateHash = Animator.StringToHash("Run");
    private static readonly int JumpStateHash = Animator.StringToHash("Jump");
    private static readonly int FlipStateHash = Animator.StringToHash("Flip");
    private static readonly int PunchStateHash = Animator.StringToHash("Punch");
    private static readonly int KickStateHash = Animator.StringToHash("Kick");
    private static readonly int SleepDownStateHash = Animator.StringToHash("SleepDown");
    private static readonly int SleepingStateHash = Animator.StringToHash("Sleeping");
    private static readonly int GetUpStateHash = Animator.StringToHash("GetUp");

    private CharacterController characterController;
    private Vector3 velocity;
    private Vector2 aiMoveInput;
    private Vector2 lastMoveInput;
    private CollisionFlags lastCollisionFlags;
    private bool wasGrounded;
    private bool hasSpeedParameter;
    private bool hasJumpParameter;
    private bool hasFlipParameter;
    private bool hasPunchParameter;
    private bool hasKickParameter;
    private bool hasSleepParameter;
    private bool hasWakeParameter;
    private bool hasGroundedParameter;
    private bool hasVerticalSpeedParameter;
    private bool aiRun;
    private bool aiSleeping;
    private bool aiWakingUp;
    private bool isFlipJumping;
    private bool hasForcedAirExit;
    private bool jumpInputConsumed;
    private float lastGroundedTime;
    private float attackLockedUntil;
    private bool attackInProgress;
    private bool attackRecoveryInProgress;
    private int activeAttackStateHash;
    private float activeAttackExitTime;
    private bool hasSleepAnchor;
    private Vector3 sleepAnchorPosition;
    private Quaternion sleepAnchorRotation;
    private float wakeMovementLockedUntil;
    private float poseRecoveryLockedUntil;

    public RobotControlMode ControlMode => controlMode;
    public bool IsAutoControlled => controlMode == RobotControlMode.Auto;
    public bool IsSleeping => aiSleeping;
    public bool IsWakingUp => aiWakingUp;
    public bool IsBodyLocked => aiSleeping || aiWakingUp || Time.time < poseRecoveryLockedUntil;
    public Vector2 LastMoveInput => lastMoveInput;
    public CollisionFlags LastCollisionFlags => lastCollisionFlags;
    public bool IsGrounded => characterController != null && characterController.isGrounded;

    private void Awake()
    {
        characterController = GetComponent<CharacterController>();
        KeepCapsuleBottomAtRoot();

        if (animator == null)
        {
            animator = GetComponent<Animator>();
        }

        if (cameraTransform == null && Camera.main != null)
        {
            cameraTransform = Camera.main.transform;
        }

        if (attackVfx == null)
        {
            attackVfx = GetComponent<AttackVfxController>();
        }

        if (attackVfx == null)
        {
            attackVfx = gameObject.AddComponent<AttackVfxController>();
        }

        CacheAnimatorParameters();
    }

    private void KeepCapsuleBottomAtRoot()
    {
        float desiredCenterY = characterController.height * 0.5f;
        if (characterController.center.y < desiredCenterY - 0.01f)
        {
            Vector3 center = characterController.center;
            center.y = desiredCenterY;
            characterController.center = center;
        }

        characterController.skinWidth = Mathf.Min(characterController.skinWidth, Mathf.Max(characterController.radius * 0.02f, 0.005f));
    }

    private void Update()
    {
        HandleControlModeHotkeys();

        bool isGrounded = characterController.isGrounded;
        Vector2 moveInput = ReadMoveInput();
        lastMoveInput = moveInput;
        bool isRunning = moveInput.sqrMagnitude > 0.0001f && ReadRunHeld();
        Vector3 moveDirection = GetCameraRelativeMove(moveInput);

        if (aiSleeping)
        {
            UpdateSleepingBody(isGrounded);
            wasGrounded = characterController.isGrounded;
            return;
        }

        if (aiWakingUp)
        {
            UpdateWakingBody(isGrounded);
            wasGrounded = characterController.isGrounded;
            return;
        }

        if (Time.time < poseRecoveryLockedUntil)
        {
            UpdatePoseRecoveryBody(isGrounded);
            wasGrounded = characterController.isGrounded;
            return;
        }

        if (isGrounded && velocity.y < 0f)
        {
            velocity.y = groundedStickVelocity;
        }

        if (EnsureLocomotionPoseUnlocked())
        {
            wasGrounded = characterController.isGrounded;
            return;
        }

        bool jumpPressed = ReadJumpPressed();
        if (isGrounded && jumpPressed && !jumpInputConsumed)
        {
            jumpInputConsumed = true;
            velocity.y = Mathf.Sqrt(jumpHeight * -2f * gravity);
            SetJumpTrigger(isRunning);
        }
        else if (!ReadJumpHeld())
        {
            jumpInputConsumed = false;
        }

        RotateToward(moveDirection);

        float currentSpeed = isRunning ? runSpeed : walkSpeed;
        Vector3 horizontalVelocity = moveDirection * currentSpeed;
        velocity.y += gravity * Time.deltaTime;
        CollisionFlags collisionFlags = characterController.Move((horizontalVelocity + Vector3.up * velocity.y) * Time.deltaTime);
        lastCollisionFlags = collisionFlags;

        bool groundedAfterMove = characterController.isGrounded || (collisionFlags & CollisionFlags.Below) != 0;
        if (groundedAfterMove)
        {
            lastGroundedTime = Time.time;
        }

        UpdateAttackRecoveryState();
        UpdateScriptedAttackState(moveInput, isRunning, groundedAfterMove);
        bool attackStarted = CanStartGroundAction(groundedAfterMove) && TryStartAttack();
        if (!wasGrounded && groundedAfterMove && !isFlipJumping)
        {
            SyncLandingAnimation();
        }

        if (isFlipJumping && HasFlipFinished())
        {
            isFlipJumping = false;
        }

        UpdateAnimator(moveInput, isRunning, groundedAfterMove);
        ExitFinishedAirAnimation(moveInput, isRunning, groundedAfterMove);
        if (!attackStarted)
        {
            ExitFinishedAttackAnimation(moveInput, isRunning, groundedAfterMove);
        }

        UpdateAttackVfx();
        wasGrounded = groundedAfterMove;
    }

    private void LateUpdate()
    {
        if ((!aiSleeping && !aiWakingUp && Time.time >= poseRecoveryLockedUntil) || !hasSleepAnchor)
        {
            return;
        }

        transform.SetPositionAndRotation(sleepAnchorPosition, sleepAnchorRotation);
    }

    private Vector2 ReadMoveInput()
    {
        Vector2 keyboardInput = ReadKeyboardMoveInput();
        if (controlMode == RobotControlMode.Auto)
        {
            if (IsBodyLocked)
            {
                return Vector2.zero;
            }

            if (keyboardInput.sqrMagnitude > 0.0001f)
            {
                if (aiSleeping)
                {
                    SetAiSleeping(false);
                }

                return keyboardInput;
            }

            return IsBodyLocked ? Vector2.zero : aiMoveInput;
        }

        return keyboardInput;
    }

    private Vector2 ReadKeyboardMoveInput()
    {
        Keyboard keyboard = Keyboard.current;
        if (keyboard == null)
        {
            return Vector2.zero;
        }

        Vector2 input = Vector2.zero;

        if (keyboard.wKey.isPressed || keyboard.upArrowKey.isPressed)
        {
            input.y += 1f;
        }

        if (keyboard.sKey.isPressed || keyboard.downArrowKey.isPressed)
        {
            input.y -= 1f;
        }

        if (keyboard.dKey.isPressed || keyboard.rightArrowKey.isPressed)
        {
            input.x += 1f;
        }

        if (keyboard.aKey.isPressed || keyboard.leftArrowKey.isPressed)
        {
            input.x -= 1f;
        }

        return Vector2.ClampMagnitude(input, 1f);
    }

    private bool ReadJumpPressed()
    {
        Keyboard keyboard = Keyboard.current;
        return keyboard != null && keyboard.spaceKey.wasPressedThisFrame;
    }

    private bool ReadJumpHeld()
    {
        Keyboard keyboard = Keyboard.current;
        return keyboard != null && keyboard.spaceKey.isPressed;
    }

    private bool ReadRunHeld()
    {
        if (controlMode == RobotControlMode.Auto && aiMoveInput.sqrMagnitude > 0.0001f)
        {
            return aiRun;
        }

        Keyboard keyboard = Keyboard.current;
        return keyboard != null && (keyboard.leftShiftKey.isPressed || keyboard.rightShiftKey.isPressed);
    }

    private bool ReadPunchPressed()
    {
        Keyboard keyboard = Keyboard.current;
        return keyboard != null && keyboard.fKey.wasPressedThisFrame;
    }

    private bool ReadPunchHeld()
    {
        Keyboard keyboard = Keyboard.current;
        return keyboard != null && keyboard.fKey.isPressed;
    }

    private bool ReadKickPressed()
    {
        Keyboard keyboard = Keyboard.current;
        return keyboard != null && keyboard.gKey.wasPressedThisFrame;
    }

    private bool ReadKickHeld()
    {
        Keyboard keyboard = Keyboard.current;
        return keyboard != null && keyboard.gKey.isPressed;
    }

    private Vector3 GetCameraRelativeMove(Vector2 input)
    {
        if (input.sqrMagnitude <= 0.0001f)
        {
            return Vector3.zero;
        }

        Vector3 forward = Vector3.forward;
        Vector3 right = Vector3.right;

        if (cameraTransform != null)
        {
            forward = cameraTransform.forward;
            right = cameraTransform.right;
            forward.y = 0f;
            right.y = 0f;
            forward.Normalize();
            right.Normalize();
        }

        return (forward * input.y + right * input.x).normalized;
    }

    private void RotateToward(Vector3 moveDirection)
    {
        if (moveDirection.sqrMagnitude <= 0.0001f)
        {
            return;
        }

        Quaternion targetRotation = Quaternion.LookRotation(moveDirection);
        float t = 1f - Mathf.Exp(-rotationSharpness * Time.deltaTime);
        transform.rotation = Quaternion.Slerp(transform.rotation, targetRotation, t);
    }

    private void UpdateAnimator(Vector2 moveInput, bool isRunning, bool isGrounded)
    {
        if (animator == null)
        {
            return;
        }

        if (hasSpeedParameter)
        {
            float animationSpeed = moveInput.magnitude <= 0.0001f ? 0f : isRunning ? 2f : 1f;
            animator.SetFloat(SpeedHash, animationSpeed);
        }

        if (hasGroundedParameter)
        {
            animator.SetBool(GroundedHash, isGrounded);
        }

        if (hasVerticalSpeedParameter)
        {
            animator.SetFloat(VerticalSpeedHash, velocity.y);
        }
    }

    private void SetJumpTrigger(bool isRunning)
    {
        if (animator == null)
        {
            return;
        }

        if (isRunning && hasFlipParameter)
        {
            isFlipJumping = true;
            hasForcedAirExit = false;
            animator.ResetTrigger(JumpHash);
            animator.SetTrigger(FlipHash);
            animator.CrossFade(FlipStateHash, 0.05f, 0, flipStartNormalizedTime);
        }
        else if (hasJumpParameter)
        {
            isFlipJumping = false;
            hasForcedAirExit = false;
            animator.ResetTrigger(FlipHash);
            animator.SetTrigger(JumpHash);
        }
    }

    private bool TryStartAttack()
    {
        if (animator == null || IsInAirActionState())
        {
            return false;
        }

        if (ReadPunchPressed() && CanStartAttackNow())
        {
            StartAttack(PunchStateHash, punchStartNormalizedTime, punchRepeatDelay);
            return true;
        }

        if (ReadKickPressed() && CanStartAttackNow())
        {
            StartAttack(KickStateHash, kickStartNormalizedTime, kickRepeatDelay);
            return true;
        }

        if (ReadPunchHeld() && CanStartAttackNow())
        {
            StartAttack(PunchStateHash, punchStartNormalizedTime, punchRepeatDelay);
            return true;
        }

        if (ReadKickHeld() && CanStartAttackNow())
        {
            StartAttack(KickStateHash, kickStartNormalizedTime, kickRepeatDelay);
            return true;
        }

        return false;
    }

    private bool CanStartGroundAction(bool isGrounded)
    {
        return isGrounded || Time.time - lastGroundedTime <= groundedActionGraceTime;
    }

    private bool CanStartAttackNow()
    {
        if (Time.time < attackLockedUntil || attackInProgress || IsTransitioningIntoAttack())
        {
            return false;
        }

        return true;
    }

    private void UpdateScriptedAttackState(Vector2 moveInput, bool isRunning, bool isGrounded)
    {
        if (!attackInProgress || Time.time < activeAttackExitTime)
        {
            return;
        }

        if (animator == null)
        {
            attackInProgress = false;
            return;
        }

        AnimatorStateInfo stateInfo = animator.GetCurrentAnimatorStateInfo(0);
        if (stateInfo.shortNameHash == activeAttackStateHash)
        {
            float exitTime = activeAttackStateHash == PunchStateHash ? punchExitNormalizedTime : kickExitNormalizedTime;
            if (stateInfo.normalizedTime < exitTime)
            {
                return;
            }
        }

        attackInProgress = false;
        if (isGrounded)
        {
            CrossFadeToLocomotionOnce(moveInput, isRunning);
        }
    }

    private bool IsAttackState(int stateHash)
    {
        return stateHash == PunchStateHash || stateHash == KickStateHash;
    }

    private bool IsTransitioningIntoAttack()
    {
        if (!animator.IsInTransition(0))
        {
            return false;
        }

        AnimatorStateInfo nextStateInfo = animator.GetNextAnimatorStateInfo(0);
        return nextStateInfo.shortNameHash == PunchStateHash || nextStateInfo.shortNameHash == KickStateHash;
    }

    private void StartAttack(int stateHash, float startNormalizedTime, float repeatDelay)
    {
        attackLockedUntil = Time.time + attackRestartLockTime;
        activeAttackStateHash = stateHash;
        activeAttackExitTime = Time.time + Mathf.Max(attackRestartLockTime, repeatDelay);
        attackInProgress = true;
        attackRecoveryInProgress = false;
        ClearActionTriggers();
        animator.Play(stateHash, 0, startNormalizedTime);
    }

    private bool IsInAirActionState()
    {
        if (animator == null)
        {
            return false;
        }

        AnimatorStateInfo stateInfo = animator.GetCurrentAnimatorStateInfo(0);
        return stateInfo.shortNameHash == JumpStateHash || stateInfo.shortNameHash == FlipStateHash;
    }

    private void ClearActionTriggers()
    {
        animator.ResetTrigger(JumpHash);
        animator.ResetTrigger(FlipHash);
        animator.ResetTrigger(PunchHash);
        animator.ResetTrigger(KickHash);
        if (hasSleepParameter)
        {
            animator.ResetTrigger(SleepHash);
        }
        if (hasWakeParameter)
        {
            animator.ResetTrigger(WakeHash);
        }
    }

    private bool HasFlipFinished()
    {
        if (animator == null)
        {
            return true;
        }

        AnimatorStateInfo stateInfo = animator.GetCurrentAnimatorStateInfo(0);
        return stateInfo.shortNameHash != FlipStateHash || stateInfo.normalizedTime >= flipExitNormalizedTime;
    }

    private void ExitFinishedAirAnimation(Vector2 moveInput, bool isRunning, bool isGrounded)
    {
        if (!isGrounded || animator == null || hasForcedAirExit)
        {
            return;
        }

        AnimatorStateInfo stateInfo = animator.GetCurrentAnimatorStateInfo(0);
        bool airStateIsActive = stateInfo.shortNameHash == JumpStateHash || stateInfo.shortNameHash == FlipStateHash;
        float exitTime = stateInfo.shortNameHash == FlipStateHash ? flipExitNormalizedTime : airAnimationExitTime;
        if (!airStateIsActive || stateInfo.normalizedTime < exitTime)
        {
            return;
        }

        ClearActionTriggers();
        CrossFadeToLocomotionOnce(moveInput, isRunning);
        isFlipJumping = false;
        hasForcedAirExit = true;
    }

    private void ExitFinishedAttackAnimation(Vector2 moveInput, bool isRunning, bool isGrounded)
    {
        if (!isGrounded || animator == null || attackRecoveryInProgress)
        {
            return;
        }

        AnimatorStateInfo stateInfo = animator.GetCurrentAnimatorStateInfo(0);
        bool punchIsActive = stateInfo.shortNameHash == PunchStateHash;
        bool kickIsActive = stateInfo.shortNameHash == KickStateHash;
        if (!punchIsActive && !kickIsActive)
        {
            return;
        }

        float exitTime = punchIsActive ? punchExitNormalizedTime : kickExitNormalizedTime;
        if (stateInfo.normalizedTime < exitTime)
        {
            return;
        }

        ClearActionTriggers();
        CrossFadeToLocomotionOnce(moveInput, isRunning);
    }

    private void CrossFadeToLocomotionOnce(Vector2 moveInput, bool isRunning)
    {
        int locomotionState = moveInput.magnitude <= 0.0001f ? IdleStateHash : isRunning ? RunStateHash : WalkStateHash;
        attackRecoveryInProgress = true;
        animator.CrossFade(locomotionState, 0.1f, 0, 0f);
    }

    private void UpdateAttackRecoveryState()
    {
        if (!attackRecoveryInProgress || animator == null)
        {
            return;
        }

        AnimatorStateInfo stateInfo = animator.GetCurrentAnimatorStateInfo(0);
        if (!IsAttackState(stateInfo.shortNameHash))
        {
            attackRecoveryInProgress = false;
        }
    }

    private void UpdateAttackVfx()
    {
        if (attackVfx == null || animator == null)
        {
            return;
        }

        AnimatorStateInfo stateInfo = animator.GetCurrentAnimatorStateInfo(0);
        bool punchIsActive = stateInfo.shortNameHash == PunchStateHash && stateInfo.normalizedTime < punchExitNormalizedTime;
        bool kickIsActive = stateInfo.shortNameHash == KickStateHash && stateInfo.normalizedTime < kickExitNormalizedTime;

        attackVfx.SetPunchActive(punchIsActive);
        attackVfx.SetKickActive(kickIsActive);
    }

    private void SyncLandingAnimation()
    {
        if (!syncLandingToJumpClip || animator == null)
        {
            return;
        }

        AnimatorStateInfo stateInfo = animator.GetCurrentAnimatorStateInfo(0);
        AnimatorStateInfo nextStateInfo = animator.GetNextAnimatorStateInfo(0);
        bool jumpIsActive = stateInfo.shortNameHash == JumpStateHash || nextStateInfo.shortNameHash == JumpStateHash;

        if (jumpIsActive && stateInfo.normalizedTime < landingClipNormalizedTime)
        {
            animator.CrossFade(JumpStateHash, 0.05f, 0, landingClipNormalizedTime);
        }
    }

    private void CacheAnimatorParameters()
    {
        if (animator == null)
        {
            return;
        }

        foreach (AnimatorControllerParameter parameter in animator.parameters)
        {
            if (parameter.nameHash == SpeedHash)
            {
                hasSpeedParameter = true;
            }
            else if (parameter.nameHash == JumpHash)
            {
                hasJumpParameter = true;
            }
            else if (parameter.nameHash == FlipHash)
            {
                hasFlipParameter = true;
            }
            else if (parameter.nameHash == PunchHash)
            {
                hasPunchParameter = true;
            }
            else if (parameter.nameHash == KickHash)
            {
                hasKickParameter = true;
            }
            else if (parameter.nameHash == SleepHash)
            {
                hasSleepParameter = true;
            }
            else if (parameter.nameHash == WakeHash)
            {
                hasWakeParameter = true;
            }
            else if (parameter.nameHash == GroundedHash)
            {
                hasGroundedParameter = true;
            }
            else if (parameter.nameHash == VerticalSpeedHash)
            {
                hasVerticalSpeedParameter = true;
            }
        }
    }

    private void HandleControlModeHotkeys()
    {
        Keyboard keyboard = Keyboard.current;
        if (keyboard == null)
        {
            return;
        }

        if (keyboard.pKey.wasPressedThisFrame)
        {
            SetControlMode(RobotControlMode.Auto);
        }
        else if (keyboard.mKey.wasPressedThisFrame)
        {
            SetControlMode(RobotControlMode.Manual);
        }
        else if (keyboard.tabKey.wasPressedThisFrame)
        {
            SetControlMode(controlMode == RobotControlMode.Auto ? RobotControlMode.Manual : RobotControlMode.Auto);
        }

        if (keyboard.zKey.wasPressedThisFrame)
        {
            SetControlMode(RobotControlMode.Auto);
            SetAiMove(Vector2.zero);
            SetAiSleeping(true);
        }
        else if (keyboard.xKey.wasPressedThisFrame)
        {
            SetControlMode(RobotControlMode.Auto);
            SetAiMove(Vector2.zero);
            SetAiSleeping(false);
        }
    }

    private void UpdateSleepingBody(bool isGrounded)
    {
        aiMoveInput = Vector2.zero;
        lastMoveInput = Vector2.zero;
        aiRun = false;
        velocity = Vector3.zero;
        lastCollisionFlags = CollisionFlags.None;

        if (hasSleepAnchor)
        {
            transform.SetPositionAndRotation(sleepAnchorPosition, sleepAnchorRotation);
        }

        UpdateAnimator(Vector2.zero, false, isGrounded);
        UpdateAttackVfx();
    }

    private void UpdateWakingBody(bool isGrounded)
    {
        aiMoveInput = Vector2.zero;
        lastMoveInput = Vector2.zero;
        aiRun = false;
        velocity = Vector3.zero;
        lastCollisionFlags = CollisionFlags.None;

        if (hasSleepAnchor)
        {
            transform.SetPositionAndRotation(sleepAnchorPosition, sleepAnchorRotation);
        }

        UpdateAnimator(Vector2.zero, false, isGrounded);
        UpdateAttackVfx();

        if (Time.time >= wakeMovementLockedUntil)
        {
            aiWakingUp = false;
            poseRecoveryLockedUntil = Time.time + Mathf.Max(0.1f, poseRecoveryLockSeconds);
            ForceAnimatorIdleReset();
        }
    }

    private void UpdatePoseRecoveryBody(bool isGrounded)
    {
        aiMoveInput = Vector2.zero;
        lastMoveInput = Vector2.zero;
        aiRun = false;
        velocity = Vector3.zero;
        lastCollisionFlags = CollisionFlags.None;

        if (hasSleepAnchor)
        {
            transform.SetPositionAndRotation(sleepAnchorPosition, sleepAnchorRotation);
        }

        UpdateAnimator(Vector2.zero, false, isGrounded);
        UpdateAttackVfx();

        if (Time.time >= poseRecoveryLockedUntil)
        {
            hasSleepAnchor = false;
            ForceAnimatorIdleReset();
        }
    }

    private bool EnsureLocomotionPoseUnlocked()
    {
        if (animator == null)
        {
            return false;
        }

        AnimatorStateInfo stateInfo = animator.GetCurrentAnimatorStateInfo(0);
        int stateHash = stateInfo.shortNameHash;
        if (stateHash == SleepDownStateHash || stateHash == SleepingStateHash || stateHash == GetUpStateHash)
        {
            aiMoveInput = Vector2.zero;
            lastMoveInput = Vector2.zero;
            aiRun = false;
            velocity = Vector3.zero;
            hasSleepAnchor = true;
            sleepAnchorPosition = transform.position;
            sleepAnchorRotation = transform.rotation;
            poseRecoveryLockedUntil = Time.time + Mathf.Max(0.1f, poseRecoveryLockSeconds);
            ForceAnimatorIdleReset();
            return true;
        }

        return false;
    }

    private void ForceAnimatorIdleReset()
    {
        if (animator == null)
        {
            return;
        }

        ClearActionTriggers();
        animator.Rebind();
        animator.Update(0f);
        animator.CrossFade(IdleStateHash, 0.05f, 0, 0f);
    }

    public void SetControlMode(RobotControlMode mode)
    {
        controlMode = mode;
        if (controlMode == RobotControlMode.Manual)
        {
            aiMoveInput = Vector2.zero;
            aiRun = false;
            SetAiSleeping(false);
        }
    }

    public void SetAiMove(Vector2 moveInput, bool run = false)
    {
        if (IsBodyLocked)
        {
            aiMoveInput = Vector2.zero;
            aiRun = false;
            return;
        }

        aiMoveInput = Vector2.ClampMagnitude(moveInput, 1f);
        aiRun = run;
    }

    public void SetAiCommand(string action, string mode = "")
    {
        SetControlMode(RobotControlMode.Auto);
        string normalized = string.IsNullOrWhiteSpace(action) ? "idle" : action.Trim().ToLowerInvariant();
        switch (normalized)
        {
            case "up":
            case "forward":
            case "move_up":
                SetAiMove(Vector2.up);
                break;
            case "down":
            case "back":
            case "backward":
            case "move_down":
                SetAiMove(Vector2.down);
                break;
            case "left":
            case "move_left":
                SetAiMove(Vector2.left);
                break;
            case "right":
            case "move_right":
                SetAiMove(Vector2.right);
                break;
            case "sleep":
                SetAiMove(Vector2.zero);
                SetAiSleeping(true);
                break;
            case "wake":
                SetAiMove(Vector2.zero);
                SetAiSleeping(false);
                break;
            case "idle":
                SetAiMove(Vector2.zero);
                if (!IsBodyLocked)
                {
                    ForceAnimatorIdleReset();
                }
                break;
            default:
                SetAiMove(Vector2.zero);
                if (!string.Equals(mode, "sleep", System.StringComparison.OrdinalIgnoreCase))
                {
                    SetAiSleeping(false);
                }
                break;
        }

        if (string.Equals(mode, "sleep", System.StringComparison.OrdinalIgnoreCase))
        {
            SetAiSleeping(true);
        }
    }

    public void SetAiSleeping(bool sleeping)
    {
        if (sleeping && aiSleeping)
        {
            return;
        }

        if (!sleeping && !aiSleeping && aiWakingUp)
        {
            return;
        }

        if (!sleeping && !aiSleeping && !aiWakingUp)
        {
            return;
        }

        aiSleeping = sleeping;
        if (sleeping)
        {
            poseRecoveryLockedUntil = 0f;
        }

        aiMoveInput = Vector2.zero;
        aiRun = false;
        velocity = Vector3.zero;
        attackInProgress = false;
        attackRecoveryInProgress = false;
        isFlipJumping = false;
        hasForcedAirExit = false;

        if (animator == null)
        {
            return;
        }

        ClearActionTriggers();
        if (sleeping)
        {
            aiWakingUp = false;
            hasSleepAnchor = true;
            sleepAnchorPosition = transform.position;
            sleepAnchorRotation = transform.rotation;
            animator.CrossFade(SleepDownStateHash, 0.12f, 0, 1f);
        }
        else
        {
            aiWakingUp = true;
            wakeMovementLockedUntil = Time.time + Mathf.Max(0.1f, getUpMovementLockSeconds);
            animator.CrossFade(GetUpStateHash, 0.12f, 0, 0f);
        }
    }

    public void Respawn(Vector3 position, Quaternion rotation)
    {
        aiSleeping = false;
        aiWakingUp = false;
        hasSleepAnchor = false;
        poseRecoveryLockedUntil = 0f;
        aiMoveInput = Vector2.zero;
        lastMoveInput = Vector2.zero;
        aiRun = false;
        velocity = Vector3.zero;
        attackInProgress = false;
        attackRecoveryInProgress = false;
        isFlipJumping = false;
        hasForcedAirExit = false;
        lastCollisionFlags = CollisionFlags.None;

        if (characterController != null)
        {
            characterController.enabled = false;
            transform.SetPositionAndRotation(position, rotation);
            characterController.enabled = true;
        }
        else
        {
            transform.SetPositionAndRotation(position, rotation);
        }

        if (animator != null)
        {
            ClearActionTriggers();
            animator.CrossFade(IdleStateHash, 0.08f, 0, 0f);
        }
    }

    public string GetCurrentAnimatorStateName()
    {
        if (animator == null)
        {
            return "none";
        }

        AnimatorStateInfo stateInfo = animator.GetCurrentAnimatorStateInfo(0);
        if (stateInfo.shortNameHash == IdleStateHash) return "Idle";
        if (stateInfo.shortNameHash == WalkStateHash) return "Walk";
        if (stateInfo.shortNameHash == RunStateHash) return "Run";
        if (stateInfo.shortNameHash == JumpStateHash) return "Jump";
        if (stateInfo.shortNameHash == FlipStateHash) return "Flip";
        if (stateInfo.shortNameHash == PunchStateHash) return "Punch";
        if (stateInfo.shortNameHash == KickStateHash) return "Kick";
        if (stateInfo.shortNameHash == SleepDownStateHash) return "SleepDown";
        if (stateInfo.shortNameHash == SleepingStateHash) return "Sleeping";
        if (stateInfo.shortNameHash == GetUpStateHash) return "GetUp";
        return stateInfo.shortNameHash.ToString();
    }
}
