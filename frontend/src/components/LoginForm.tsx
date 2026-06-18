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
        <h1 className="text-3xl font-bold tracking-tight text-white sm:text-4xl">
          旅行助手
        </h1>
        <p className="mt-2 text-sm text-slate-400">
          AI 驱动的智能旅行规划平台
        </p>
      </div>

      {/* Card */}
      <div className="relative overflow-hidden rounded-2xl border border-white/10 bg-slate-900/60 p-6 shadow-2xl backdrop-blur-xl sm:p-8">
        {/* ambient glow */}
        <div className="pointer-events-none absolute -right-20 -top-20 h-40 w-40 rounded-full bg-blue-500/20 blur-3xl" />
        <div className="pointer-events-none absolute -bottom-20 -left-20 h-40 w-40 rounded-full bg-indigo-500/10 blur-3xl" />

        <h2 className="relative mb-6 text-center text-xl font-semibold text-white">
          欢迎回来
        </h2>

        {/* Role tabs */}
        <div className="relative mb-6 grid grid-cols-3 gap-2 rounded-xl bg-slate-800/60 p-1">
          {roles.map((r) => (
            <button
              key={r.id}
              type="button"
              onClick={() => setRole(r.id)}
              className={cn(
                "relative z-10 flex items-center justify-center gap-1.5 rounded-lg px-2 py-2.5 text-xs font-medium transition-all duration-200 sm:text-sm",
                role === r.id
                  ? r.id === "vip"
                    ? "text-amber-950"
                    : r.id === "admin"
                      ? "text-white"
                      : "text-white"
                  : "text-slate-400 hover:text-slate-200"
              )}
            >
              {r.icon}
              {r.label}
              {role === r.id && (
                <span
                  className={cn(
                    "absolute inset-0 -z-10 rounded-lg transition-all duration-200",
                    r.id === "vip" &&
                      "bg-gradient-to-r from-amber-300 via-yellow-400 to-amber-500 shadow-lg shadow-amber-500/25",
                    r.id === "admin" &&
                      "bg-gradient-to-r from-indigo-500 to-violet-600 shadow-lg shadow-indigo-500/25",
                    r.id === "user" && "bg-slate-700 shadow-lg shadow-slate-900/40"
                  )}
                />
              )}
            </button>
          ))}
        </div>

        {error && (
          <div className="mb-4 rounded-lg border border-red-500/20 bg-red-500/10 px-4 py-3 text-sm text-red-200">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="relative space-y-5">
          {/* Email */}
          <div className="space-y-1.5">
            <label
              htmlFor="email"
              className="block text-sm font-medium text-slate-300"
            >
              邮箱地址
            </label>
            <div className="relative">
              <Mail className="pointer-events-none absolute left-3 top-1/2 h-5 w-5 -translate-y-1/2 text-slate-500" />
              <input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="name@example.com"
                autoComplete="email"
                className="w-full rounded-xl border border-slate-700 bg-slate-950/50 py-2.5 pl-10 pr-4 text-sm text-white placeholder:text-slate-600 outline-none transition-all duration-200 focus:border-blue-500 focus:ring-4 focus:ring-blue-500/20"
              />
            </div>
          </div>

          {/* Password */}
          <div className="space-y-1.5">
            <label
              htmlFor="password"
              className="block text-sm font-medium text-slate-300"
            >
              密码
            </label>
            <div className="relative">
              <Lock className="pointer-events-none absolute left-3 top-1/2 h-5 w-5 -translate-y-1/2 text-slate-500" />
              <input
                id="password"
                type={showPassword ? "text" : "password"}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="至少 6 位字符"
                autoComplete="current-password"
                className="w-full rounded-xl border border-slate-700 bg-slate-950/50 py-2.5 pl-10 pr-10 text-sm text-white placeholder:text-slate-600 outline-none transition-all duration-200 focus:border-blue-500 focus:ring-4 focus:ring-blue-500/20"
              />
              <button
                type="button"
                onClick={() => setShowPassword((v) => !v)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 transition-colors hover:text-slate-300"
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
              "group relative w-full overflow-hidden rounded-xl py-2.5 text-sm font-semibold text-white transition-all duration-200",
              "bg-gradient-to-r from-blue-600 to-indigo-600 shadow-lg shadow-blue-600/25",
              "hover:scale-[1.02] hover:shadow-blue-600/40 active:scale-[0.98]",
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
            <span className="absolute inset-0 -translate-x-full bg-gradient-to-r from-transparent via-white/20 to-transparent transition-transform duration-700 group-hover:translate-x-full" />
          </button>
        </form>

        {/* Extra links */}
        <div className="relative mt-6 flex flex-col items-center gap-3 text-sm">
          <Link
            href="/register"
            className="text-slate-400 transition-colors hover:text-white"
          >
            还没有账号？<span className="text-blue-400 hover:text-blue-300">立即注册</span>
          </Link>

          <div className="flex items-center gap-2">
            <div className="h-px w-12 bg-slate-700" />
            <span className="text-xs text-slate-500">或</span>
            <div className="h-px w-12 bg-slate-700" />
          </div>

          <button
            type="button"
            onClick={handleGuest}
            disabled={guestLoading}
            className="flex items-center gap-1.5 text-slate-400 transition-colors hover:text-white disabled:opacity-60"
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
      <p className="mt-6 text-center text-xs text-slate-500">
        登录即表示您同意我们的服务条款与隐私政策
      </p>
    </div>
  );
}
