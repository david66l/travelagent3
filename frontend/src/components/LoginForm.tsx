"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import {
  Mail,
  Lock,
  Eye,
  EyeOff,
  Loader2,
  Sparkles,
  Shield,
  User,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { ensureGuestSession, loginUser } from "@/lib/api";

type Role = "user" | "vip" | "admin";

interface RoleOption {
  id: Role;
  label: string;
  icon: React.ReactNode;
}

const roles: RoleOption[] = [
  {
    id: "user",
    label: "普通用户",
    icon: <User className="h-4 w-4" />,
  },
  {
    id: "vip",
    label: "VIP 用户",
    icon: <Sparkles className="h-4 w-4" />,
  },
  {
    id: "admin",
    label: "管理员",
    icon: <Shield className="h-4 w-4" />,
  },
];

export default function LoginForm() {
  const router = useRouter();
  const [role, setRole] = useState<Role>("user");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [guestLoading, setGuestLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const validateEmail = (value: string) => {
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value);
  };

  const validateForm = () => {
    if (!email.trim()) {
      setError("请输入邮箱");
      return false;
    }
    if (!validateEmail(email)) {
      setError("邮箱格式不正确");
      return false;
    }
    if (!password) {
      setError("请输入密码");
      return false;
    }
    if (password.length < 6) {
      setError("密码至少 6 位");
      return false;
    }
    return true;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (!validateForm()) return;

    setIsLoading(true);
    try {
      const data = await loginUser({ email, password });
      const targetRole = data.role || role;
      if (targetRole === "admin") {
        router.push("/admin");
      } else {
        router.push("/chat");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "登录失败，请重试");
    } finally {
      setIsLoading(false);
    }
  };

  const handleGuest = async () => {
    setGuestLoading(true);
    setError(null);
    try {
      await ensureGuestSession();
      router.push("/chat");
    } catch (err) {
      setError(err instanceof Error ? err.message : "游客模式进入失败");
    } finally {
      setGuestLoading(false);
    }
  };

  return (
    <div className="w-full max-w-md">
      {/* Logo / Title */}
      <div className="mb-8 text-center">
        <h1 className="text-3xl font-bold tracking-tight text-ink sm:text-4xl">
          旅行助手
        </h1>
        <p className="mt-2 text-sm text-mute">
          AI 驱动的智能旅行规划平台
        </p>
      </div>

      {/* Card */}
      <div className="glass-panel relative overflow-hidden p-6 sm:p-8">
        <h2 className="relative mb-6 text-center text-xl font-semibold text-ink">
          欢迎回来
        </h2>

        {/* Role tabs */}
        <div className="relative mb-6 grid grid-cols-3 gap-2 rounded-xl bg-canvas-soft p-1">
          {roles.map((r) => (
            <button
              key={r.id}
              type="button"
              onClick={() => setRole(r.id)}
              className={cn(
                "relative z-10 flex items-center justify-center gap-1.5 rounded-lg px-2 py-2.5 text-xs font-medium transition-all duration-200 sm:text-sm",
                role === r.id
                  ? "bg-ink text-canvas"
                  : "text-body hover:text-ink hover:bg-canvas"
              )}
            >
              {r.icon}
              {r.label}
            </button>
          ))}
        </div>

        {error && (
          <div className="mb-4 rounded-lg border border-negative/20 bg-negative/10 px-4 py-3 text-sm text-negative-deep">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="relative space-y-5">
          {/* Email */}
          <div className="space-y-1.5">
            <label
              htmlFor="email"
              className="block text-sm font-medium text-body"
            >
              邮箱地址
            </label>
            <div className="relative">
              <Mail className="pointer-events-none absolute left-3 top-1/2 h-5 w-5 -translate-y-1/2 text-mute" />
              <input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="name@example.com"
                autoComplete="email"
                className="input-wise w-full py-2.5 pl-10 pr-4"
              />
            </div>
          </div>

          {/* Password */}
          <div className="space-y-1.5">
            <label
              htmlFor="password"
              className="block text-sm font-medium text-body"
            >
              密码
            </label>
            <div className="relative">
              <Lock className="pointer-events-none absolute left-3 top-1/2 h-5 w-5 -translate-y-1/2 text-mute" />
              <input
                id="password"
                type={showPassword ? "text" : "password"}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="至少 6 位字符"
                autoComplete="current-password"
                className="input-wise w-full py-2.5 pl-10 pr-10"
              />
              <button
                type="button"
                onClick={() => setShowPassword((v) => !v)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-mute transition-colors hover:text-ink"
                tabIndex={-1}
              >
                {showPassword ? (
                  <EyeOff className="h-5 w-5" />
                ) : (
                  <Eye className="h-5 w-5" />
                )}
              </button>
            </div>
          </div>

          {/* Submit */}
          <button
            type="submit"
            disabled={isLoading}
            className={cn(
              "btn-primary-dark group relative w-full overflow-hidden py-2.5 text-sm transition-all duration-200",
              isLoading && "cursor-not-allowed opacity-80"
            )}
          >
            <span className="relative z-10 flex items-center justify-center gap-2">
              {isLoading ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  登录中...
                </>
              ) : (
                "登录"
              )}
            </span>
          </button>
        </form>

        {/* Extra links */}
        <div className="relative mt-6 flex flex-col items-center gap-3 text-sm">
          <Link
            href="/register"
            className="text-body transition-colors hover:text-ink"
          >
            还没有账号？<span className="text-ink hover:text-primary">立即注册</span>
          </Link>

          <div className="flex items-center gap-2">
            <div className="h-px w-12 bg-hairline" />
            <span className="text-xs text-mute">或</span>
            <div className="h-px w-12 bg-hairline" />
          </div>

          <button
            type="button"
            onClick={handleGuest}
            disabled={guestLoading}
            className="flex items-center gap-1.5 text-mute transition-colors hover:text-ink disabled:opacity-60"
          >
            {guestLoading ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <Sparkles className="h-3.5 w-3.5" />
            )}
            游客模式体验
          </button>
        </div>
      </div>

      {/* Footer */}
      <p className="mt-6 text-center text-xs text-mute">
        登录即表示您同意我们的服务条款与隐私政策
      </p>
    </div>
  );
}
