import LoginForm from "@/components/LoginForm";

export const metadata = {
  title: "登录 - 旅行助手",
  description: "登录旅行助手，开启 AI 智能旅行规划",
};

export default function LoginPage() {
  return (
    <main className="fixed inset-0 z-0 flex min-h-screen w-full items-center justify-center bg-slate-950 px-4 py-12">
      {/* Background gradient */}
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-slate-800 via-slate-950 to-black" />
      <div className="pointer-events-none absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20" />

      {/* Floating orbs */}
      <div className="pointer-events-none absolute left-1/4 top-1/4 h-64 w-64 rounded-full bg-blue-600/10 blur-3xl" />
      <div className="pointer-events-none absolute bottom-1/4 right-1/4 h-72 w-72 rounded-full bg-indigo-600/10 blur-3xl" />
      <div className="pointer-events-none absolute right-1/3 top-1/2 h-48 w-48 rounded-full bg-violet-600/10 blur-3xl" />

      {/* Content */}
      <div className="relative z-10 flex w-full justify-center">
        <LoginForm />
      </div>
    </main>
  );
}
