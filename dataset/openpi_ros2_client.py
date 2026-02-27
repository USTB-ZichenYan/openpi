#!/usr/bin/env python3
"""
OpenPI ROS2 客户端（离线部署示例）

功能：
- 订阅机器人相关的 ROS2 topics（相机图像 + 电机/手部状态）
- 组装为 OpenPI 所需的观测格式，通过 WebSocket 连接策略服务器进行推理
- 将推理得到的动作结果发布为 ROS2 topic，供下游控制模块使用

参考：
- 自定义抓取机器人客户端：old_client.py（状态/动作维度定义、OpenPI 调用方式）
- OpenPI 官方离线客户端示例：offline_client.py（image_tools + WebsocketClientPolicy 用法）
- ROS2 订阅与 socket 客户端示例：robot_client_ros2.py（topic 定义与数据结构）
"""

import os
import time
import logging
from typing import Optional, Dict, Any

import numpy as np

from openpi_client import image_tools
from openpi_client import websocket_client_policy

# 在导入 rclpy 之前控制 ROS2 日志级别，避免过多输出
if "RCUTILS_LOGGING_SEVERITY" not in os.environ:
    os.environ["RCUTILS_LOGGING_SEVERITY"] = "WARN"

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, JointState
    from std_msgs.msg import Float32MultiArray
    import cv2
    from cv_bridge import CvBridge

    # 尝试导入与 robot_client_ros2.py 一致的自定义电机状态消息
    try:
        from bodyctrl_msgs.msg import MotorStatusMsg  # type: ignore

        CUSTOM_MSGS_AVAILABLE = True
    except ImportError:
        MotorStatusMsg = None  # type: ignore
        CUSTOM_MSGS_AVAILABLE = False

    ROS2_AVAILABLE = True
except ImportError as e:  # pragma: no cover - 环境问题
    print(f"[openpi_ros2_client] ROS2 相关依赖导入失败: {e}")
    ROS2_AVAILABLE = False
    CUSTOM_MSGS_AVAILABLE = False


log_level = os.environ.get("ROBOT_CLIENT_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="[%(asctime)s] [%(levelname)s] %(message)s"
)
logger = logging.getLogger("openpi_ros2_client")


class OpenPIRos2Client(Node):
    """
    OpenPI ROS2 客户端节点

    - 状态维度: 26 (与 old_client.py 中 GrabRobotClient 一致)
    - 动作维度: 26 (只使用 OpenPI 输出的前 26 维)
    - 相机: 订阅 `/camera/color/image_raw`

    订阅（对齐 robot_client_ros2.py）：
    - 图像: `/camera/color/image_raw` (sensor_msgs/Image)
    - 手部状态: `/inspire_hand/state/left_hand`, `/inspire_hand/state/right_hand` (sensor_msgs/JointState)
    - 电机状态: `/head/status`, `/waist/status`, `/arm/status`, `/leg/status` (bodyctrl_msgs/MotorStatusMsg，可选)

    发布：
    - 推理动作: `/openpi/action` (std_msgs/Float32MultiArray, 长度 26)
    """

    def __init__(
        self,
        policy_host: str = "localhost",
        policy_port: int = 8000,
        control_frequency: float = 10.0,
        action_horizon: int = 10,
        prompt: str = "pick up the box",
    ):
        super().__init__("openpi_ros2_client")

        if not ROS2_AVAILABLE:
            raise RuntimeError("ROS2 不可用，无法启动 OpenPIRos2Client")

        # OpenPI/策略服务器配置
        self.policy_host = policy_host
        self.policy_port = policy_port
        self.control_frequency = control_frequency
        self.control_period = 1.0 / control_frequency
        self.action_horizon = action_horizon
        self.prompt = prompt

        # 机器人配置（保持与 old_client.py 中 GrabRobotClient 一致）
        self.state_dim = 26
        self.action_dim = 26
        self.image_height = 224
        self.image_width = 224

        # OpenPI 策略客户端
        self.policy_client: Optional[websocket_client_policy.WebsocketClientPolicy] = None

        # ROS2 相关
        self.bridge = CvBridge()

        # 最新观测缓存
        self.latest_image: Optional[np.ndarray] = None
        self.latest_left_hand: Optional[JointState] = None
        self.latest_right_hand: Optional[JointState] = None
        # 电机状态（与 robot_client_ros2 对齐，主要使用 /arm/status）
        self.latest_head_status: Optional[Any] = None
        self.latest_waist_status: Optional[Any] = None
        self.latest_arm_status: Optional[Any] = None
        self.latest_leg_status: Optional[Any] = None

        # 发布推理动作的 topic
        self.action_pub = self.create_publisher(
            Float32MultiArray, "/openpi/action", 10
        )

        # 订阅 ROS2 topics（参考 robot_client_ros2.py 中的定义）
        self._create_subscriptions()

        # 初始化 OpenPI 策略客户端（参考 old_client.py + offline_client.py）
        self._initialize_policy_client()

        # 控制循环定时器
        self.timer = self.create_timer(self.control_period, self.control_loop)

        logger.info(
            f"OpenPIRos2Client 初始化完成, policy server: {self.policy_host}:{self.policy_port}, "
            f"freq={self.control_frequency}Hz, prompt='{self.prompt}'"
        )

    # --------------------------------------------------------------------- #
    # ROS2 topic 订阅
    # --------------------------------------------------------------------- #
    def _create_subscriptions(self):
        """创建所有需要的订阅"""
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

        qos_profile = QoSProfile(
            depth=500,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
        )

        # 相机图像
        self.create_subscription(
            Image,
            "/camera/color/image_raw",
            self.image_callback,
            qos_profile,
        )

        # 手部状态 JointState（左手）
        self.create_subscription(
            JointState,
            "/inspire_hand/state/left_hand",
            self.left_hand_callback,
            qos_profile,
        )

        # 手部状态 JointState（右手）
        self.create_subscription(
            JointState,
            "/inspire_hand/state/right_hand",
            self.right_hand_callback,
            qos_profile,
        )

        # 电机状态（只有在自定义消息可用时才订阅）
        if CUSTOM_MSGS_AVAILABLE and MotorStatusMsg is not None:
            # 与 robot_client_ros2.py 中 topics 一致
            self.create_subscription(
                MotorStatusMsg,
                "/head/status",
                self._make_motor_status_callback("head"),
                qos_profile,
            )
            self.create_subscription(
                MotorStatusMsg,
                "/waist/status",
                self._make_motor_status_callback("waist"),
                qos_profile,
            )
            self.create_subscription(
                MotorStatusMsg,
                "/arm/status",
                self._make_motor_status_callback("arm"),
                qos_profile,
            )
            self.create_subscription(
                MotorStatusMsg,
                "/leg/status",
                self._make_motor_status_callback("leg"),
                qos_profile,
            )
            logger.info(
                "已订阅: /camera/color/image_raw, /inspire_hand/state/left_hand, "
                "/inspire_hand/state/right_hand, /head/status, /waist/status, /arm/status, /leg/status"
            )
        else:
            logger.warning(
                "未能导入 bodyctrl_msgs/MotorStatusMsg，仅订阅手部和相机 topic，"
                "状态向量中的 14 维臂部关节将暂时置零"
            )
            logger.info(
                "已订阅: /camera/color/image_raw, /inspire_hand/state/left_hand, /inspire_hand/state/right_hand"
            )

    def image_callback(self, msg: Image):
        """图像回调：缓存最新图像（BGR numpy 数组）"""
        try:
            # 优先尝试 bgr8，如果失败再根据编码判断
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            except Exception:
                if msg.encoding == "rgb8":
                    cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
                else:
                    logger.warning(f"不支持的图像编码: {msg.encoding}")
                    return

            self.latest_image = cv_image
        except Exception as e:
            logger.warning(f"图像回调处理失败: {e}")

    def left_hand_callback(self, msg: JointState):
        """左手状态回调"""
        self.latest_left_hand = msg

    def right_hand_callback(self, msg: JointState):
        """右手状态回调"""
        self.latest_right_hand = msg

    # -------- 电机状态回调（与 robot_client_ros2.py 中抽取的信息对齐） -------- #
    def _make_motor_status_callback(self, part: str):
        """
        生成电机状态回调函数。

        Args:
            part: 'head' | 'waist' | 'arm' | 'leg'
        """

        def _callback(msg):
            if part == "head":
                self.latest_head_status = msg
            elif part == "waist":
                self.latest_waist_status = msg
            elif part == "arm":
                self.latest_arm_status = msg
            elif part == "leg":
                self.latest_leg_status = msg

        return _callback

    # --------------------------------------------------------------------- #
    # OpenPI 策略客户端初始化与推理
    # --------------------------------------------------------------------- #
    def _initialize_policy_client(self):
        """初始化 OpenPI WebSocket 策略客户端"""
        try:
            self.policy_client = websocket_client_policy.WebsocketClientPolicy(
                host=self.policy_host,
                port=self.policy_port,
            )
            metadata = self.policy_client.get_server_metadata()
            logger.info(f"已连接 OpenPI 策略服务器, metadata={metadata}")
        except Exception as e:
            logger.error(f"连接 OpenPI 策略服务器失败: {e}")
            raise

    def build_state_vector(self) -> np.ndarray:
        """
        构造 26 维状态向量

        说明：
        - 为了与 demension.json / old_client.py 中的定义兼容，这里使用 26 维状态，
          且精确按如下顺序和来源填充（只取对应 name 下的 pos）：
            0  left_shoulder_pitch   /arm/status name: 11
            1  left_shoulder_roll    /arm/status name: 12
            2  left_shoulder_yaw     /arm/status name: 13
            3  left_elbow_pitch      /arm/status name: 14
            4  left_wrist_yaw        /arm/status name: 15
            5  left_wrist_pitch      /arm/status name: 16
            6  left_wrist_roll       /arm/status name: 17
            7  left_little_finger    /inspire_hand/state/left_hand  name: '1'
            8  left_ring_finger      /inspire_hand/state/left_hand  name: '2'
            9  left_middle_finger    /inspire_hand/state/left_hand  name: '3'
           10  left_fore_finger      /inspire_hand/state/left_hand  name: '4'
           11  left_thumb_bend       /inspire_hand/state/left_hand  name: '5'
           12  left_thumb_rotation   /inspire_hand/state/left_hand  name: '6'
           13  right_shoulder_pitch  /arm/status name: 21
           14  right_shoulder_roll   /arm/status name: 22
           15  right_shoulder_yaw    /arm/status name: 23
           16  right_elbow_pitch     /arm/status name: 24
           17  right_wrist_yaw       /arm/status name: 25
           18  right_wrist_pitch     /arm/status name: 26
           19  right_wrist_roll      /arm/status name: 27
           20  right_little_finger   /inspire_hand/state/right_hand name: '1'
           21  right_ring_finger     /inspire_hand/state/right_hand name: '2'
           22  right_middle_finger   /inspire_hand/state/right_hand name: '3'
           23  right_fore_finger     /inspire_hand/state/right_hand name: '4'
           24  right_thumb_bend      /inspire_hand/state/right_hand name: '5'
           25  right_thumb_rotation  /inspire_hand/state/right_hand name: '6'
        """
        state = np.zeros(self.state_dim, dtype=np.float32)

        # ----------------- 利用 /arm/status 填充左右臂 14 维 ----------------- #
        if self.latest_arm_status is not None and hasattr(self.latest_arm_status, "status"):
            try:
                # 将 status 转为 name->pos 的字典，name 为 int
                name_to_pos: Dict[int, float] = {}
                for item in self.latest_arm_status.status:
                    try:
                        jid = int(item.name)
                        name_to_pos[jid] = float(item.pos)
                    except Exception:
                        continue

                # 左臂 7 维：11~17
                for offset, jid in enumerate(range(11, 18)):
                    if jid in name_to_pos:
                        state[offset] = name_to_pos[jid]

                # 右臂 7 维：21~27
                for offset, jid in enumerate(range(21, 28)):
                    if jid in name_to_pos:
                        state[13 + offset] = name_to_pos[jid]
            except Exception as e:
                logger.warning(f"从 /arm/status 解析关节位置失败，将保持臂部 14 维为 0: {e}")

        # ----------------- 左手 6 维：name '1'~'6' 的 position ----------------- #
        if self.latest_left_hand is not None:
            try:
                # name 是字符串数组，如 ['1','2',...]
                name_to_idx: Dict[str, int] = {str(n): i for i, n in enumerate(self.latest_left_hand.name)}
                for k in range(1, 7):
                    key = str(k)
                    if key in name_to_idx:
                        idx = name_to_idx[key]
                        if idx < len(self.latest_left_hand.position):
                            state[6 + k] = float(self.latest_left_hand.position[idx])
            except Exception as e:
                logger.warning(f"从 /inspire_hand/state/left_hand 解析手指位置失败，将保持左手 6 维为 0: {e}")

        # ----------------- 右手 6 维：name '1'~'6' 的 position ----------------- #
        if self.latest_right_hand is not None:
            try:
                name_to_idx: Dict[str, int] = {str(n): i for i, n in enumerate(self.latest_right_hand.name)}
                for k in range(1, 7):
                    key = str(k)
                    if key in name_to_idx:
                        idx = name_to_idx[key]
                        if idx < len(self.latest_right_hand.position):
                            state[19 + k] = float(self.latest_right_hand.position[idx])
            except Exception as e:
                logger.warning(f"从 /inspire_hand/state/right_hand 解析手指位置失败，将保持右手 6 维为 0: {e}")

        return state

    def build_observation(self) -> Optional[Dict[str, Any]]:
        """构造 OpenPI 所需的观测字典"""
        if self.latest_image is None:
            return None

        state = self.build_state_vector()

        # 预处理图像：调整为 224x224, uint8
        processed_image = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(self.latest_image, self.image_height, self.image_width)
        )

        # 从调试日志可以确认，服务器端的 mylerobot_policy 在 transform 之后
        # 直接访问 data["images"]["cam_high"] 和 data["state"]。
        # 当前配置下，input_transform 没有把 "observation.*" 的键转换为这些字段，
        # 因此这里直接按策略预期构造顶层 "state" 和 "images"。
        observation = {
            "state": state,
            "images": {
                "cam_high": processed_image,
            },
            "prompt": self.prompt,
        }

        # 调试信息：只在 DEBUG 级别打印，避免刷屏
        logger.debug(
            "Built observation keys: %s; state.shape=%s, cam_high.shape=%s",
            list(observation.keys()),
            getattr(state, "shape", None),
            getattr(processed_image, "shape", None),
        )

        return observation

    def infer_actions(self, observation: Dict[str, Any]) -> Optional[np.ndarray]:
        """调用 OpenPI 策略服务器进行推理，返回 (action_horizon, action_dim) 的动作序列"""
        if self.policy_client is None:
            logger.error("policy_client 未初始化")
            return None

        try:
            # 额外调试输出：确认在发送前关键字段是否存在
            if logger.isEnabledFor(logging.DEBUG):
                has_state = "state" in observation
                has_img = "images" in observation and isinstance(observation["images"], dict) and "cam_high" in observation["images"]
                logger.debug(
                    "Sending to policy server: has_state=%s, has_cam_high=%s",
                    has_state,
                    has_img,
                )

            t0 = time.time()
            result = self.policy_client.infer(observation)
            infer_time = time.time() - t0

            if "actions" not in result:
                logger.error(f"推理结果不包含 'actions' 字段: {result.keys()}")
                return None

            actions = result["actions"]
            actions = np.asarray(actions, dtype=np.float32)

            # 只保留前 self.action_dim 维
            if actions.ndim == 2:
                actions = actions[:, : self.action_dim]
            elif actions.ndim == 1:
                actions = actions[: self.action_dim][None, :]

            logger.debug(f"OpenPI 推理完成, 耗时 {infer_time:.3f}s, actions.shape={actions.shape}")
            return actions
        except Exception as e:
            logger.error(f"OpenPI 推理失败: {e}")
            return None

    # --------------------------------------------------------------------- #
    # 控制循环：周期性构造观测 -> 推理 -> 发布动作
    # --------------------------------------------------------------------- #
    def control_loop(self):
        """主控制循环（由 ROS2 timer 周期性调用）"""
        obs = self.build_observation()
        if obs is None:
            # 等待图像/状态就绪
            return

        actions = self.infer_actions(obs)
        if actions is None or actions.size == 0:
            return

        # 示例：只发布第 0 步动作作为当前控制命令
        first_action = actions[0]
        msg = Float32MultiArray()
        msg.data = first_action.tolist()
        self.action_pub.publish(msg)

        logger.info(
            f"发布 OpenPI 动作到 /openpi/action, 维度={len(msg.data)}, "
            f"示例前 4 维: {msg.data[:4]}"
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="OpenPI ROS2 客户端（订阅 ROS2 topics + 调用策略服务器 + 发布动作）")
    parser.add_argument("--policy-host", type=str, default="localhost", help="OpenPI 策略服务器地址")
    parser.add_argument("--policy-port", type=int, default=8000, help="OpenPI 策略服务器端口")
    parser.add_argument("--control-frequency", type=float, default=10.0, help="控制频率 (Hz)")
    parser.add_argument("--prompt", type=str, default="pick up the box", help="任务提示词")

    args = parser.parse_args()

    if not ROS2_AVAILABLE:
        logger.error("ROS2 未安装或导入失败，无法运行本客户端")
        return

    rclpy.init()
    node: Optional[OpenPIRos2Client] = None

    try:
        node = OpenPIRos2Client(
            policy_host=args.policy_host,
            policy_port=args.policy_port,
            control_frequency=args.control_frequency,
            prompt=args.prompt,
        )
        rclpy.spin(node)
    except KeyboardInterrupt:
        logger.info("收到 Ctrl+C，中止运行")
    except Exception as e:
        logger.error(f"OpenPI ROS2 客户端运行异常: {e}")
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()


