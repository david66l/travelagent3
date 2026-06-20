import LoginForm from "@/components/LoginForm";

export const metadata = {
  title: "登录 - 旅行助手",
  description: "登录旅行助手，开启 AI 智能旅行规划",
};

export default function LoginPage() {
  return (
    <main className="fixed inset-0 z-0 flex min-h-screen w-full items-center justify-center bg-canvas-soft px-4 py-12">
      {/* Content */}
      <div className="relative z-10 flex w-full justify-center">
        <LoginForm />
      </div>
    </main>
  );
}
