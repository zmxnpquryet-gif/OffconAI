plugins {
  base
}

val isWindows = System.getProperty("os.name").lowercase().contains("win")
val cargoExecutable = if (isWindows) "cargo.exe" else "cargo"
val cmakeExecutable = if (isWindows) "cmake.exe" else "cmake"
val binaryName = if (isWindows) "offcon-desktop.exe" else "offcon-desktop"
val llamaServerName = if (isWindows) "llama-server.exe" else "llama-server"

val llamaCppSourceDir = rootProject.layout.projectDirectory.dir("app/src/main/cpp/llama.cpp").asFile
val llamaCppBuildDir = layout.buildDirectory.dir("llama-cpp").get().asFile
val llamaServerBuiltFile = if (isWindows) {
  File(llamaCppBuildDir, "bin/Release/$llamaServerName")
} else {
  File(llamaCppBuildDir, "bin/$llamaServerName")
}

val cargoTest = tasks.register<Exec>("cargoTest") {
  group = "verification"
  description = "Runs Rust unit and transport tests for OffconAI Desktop."
  workingDir = projectDir
  commandLine(cargoExecutable, "test")
}

val cargoBuildRelease = tasks.register<Exec>("cargoBuildRelease") {
  group = "build"
  description = "Builds the native Rust + Slint release binary for OffconAI Desktop."
  mustRunAfter(cargoTest)
  workingDir = projectDir
  commandLine(cargoExecutable, "build", "--release")
}

val configureLlamaServer = tasks.register<Exec>("configureLlamaServer") {
  group = "build"
  description = "Configures CMake for the bundled llama-server inference engine."
  val sourceDir = llamaCppSourceDir
  val buildDir = llamaCppBuildDir
  doFirst {
    require(File(sourceDir, "CMakeLists.txt").isFile) {
      "llama.cpp submodule is missing at ${sourceDir.absolutePath}. Run `git submodule update --init --recursive`."
    }
  }
  val args = mutableListOf(
    cmakeExecutable,
    "-S", sourceDir.absolutePath,
    "-B", buildDir.absolutePath,
    "-DBUILD_SHARED_LIBS=OFF",
    "-DGGML_OPENMP=OFF",
    "-DLLAMA_OPENSSL=OFF",
    "-DLLAMA_BUILD_TESTS=OFF",
    "-DLLAMA_BUILD_EXAMPLES=OFF",
    "-DLLAMA_BUILD_TOOLS=ON",
    "-DLLAMA_BUILD_SERVER=ON",
    "-DLLAMA_BUILD_APP=OFF",
    "-DLLAMA_BUILD_UI=OFF",
    "-DLLAMA_USE_PREBUILT_UI=OFF",
    "-DGGML_NATIVE=OFF",
    "-DGGML_SSE42=ON",
    "-DGGML_AVX=ON",
    "-DGGML_AVX2=ON",
  )
  if (isWindows) {
    args += "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded"
  } else {
    args += "-DCMAKE_BUILD_TYPE=Release"
  }
  commandLine(args)
}

val buildLlamaServer = tasks.register<Exec>("buildLlamaServer") {
  group = "build"
  description = "Builds the bundled llama-server inference engine from app/src/main/cpp/llama.cpp."
  dependsOn(configureLlamaServer)
  val args = mutableListOf(
    cmakeExecutable,
    "--build", llamaCppBuildDir.absolutePath,
    "--target", "llama-server",
    "--parallel",
  )
  if (isWindows) {
    args += listOf("--config", "Release")
  }
  commandLine(args)
}

val installDist = tasks.register<Copy>("installDist") {
  group = "distribution"
  description = "Stages offcon-desktop and bundled llama-server into build/install/desktop/bin."
  dependsOn(cargoBuildRelease, buildLlamaServer)
  from(layout.projectDirectory.file("target/release/$binaryName"))
  from(llamaServerBuiltFile)
  into(layout.buildDirectory.dir("install/desktop/bin"))
}

val runDesktop = tasks.register<Exec>("run") {
  group = "application"
  description = "Runs the Rust + Slint desktop workspace."
  workingDir = projectDir
  commandLine(cargoExecutable, "run")
}

tasks.register("test") {
  group = "verification"
  description = "Alias for cargoTest."
  dependsOn(cargoTest)
}

tasks.named("check") {
  dependsOn("test")
}

tasks.named("assemble") {
  dependsOn(installDist)
}
