ThisBuild / organization := "ch.unibas.cs.gravis"
ThisBuild / version := "0.1-SNAPSHOT"
ThisBuild / scalaVersion := "3.2.0"

ThisBuild / homepage := Some(url("https://github.com/marcelluethi/inferM"))
ThisBuild / licenses += ("Apache-2.0", url(
  "http://www.apache.org/licenses/LICENSE-2.0"
))
ThisBuild / scmInfo := Some(
  ScmInfo(
    url("https://github.com/marcelluethi/monadbayes"),
    "git@github.com:marcelluethi/inferM.git"
  )
)
ThisBuild / developers := List(
  Developer(
    "marcelluethi",
    "marcelluethi",
    "marcel.luethi@unibas.ch",
    url("https://github.com/marcelluethi")
  )
)
ThisBuild / versionScheme := Some("early-semver")

lazy val root = (project in file("."))
  .settings(
    name := """inferM""",
    publishMavenStyle := true,
    publishTo := Some(
      if (isSnapshot.value)
        Opts.resolver.sonatypeSnapshots
      else
        Opts.resolver.sonatypeStaging
    ),
    resolvers ++= Seq(
      Resolver.jcenterRepo,
      Resolver.sonatypeRepo("releases"),
      Resolver.sonatypeRepo("snapshots")
    ),
    scalacOptions ++= Seq(
          "-encoding",
          "UTF-8",
          "-Xlint",
          "-deprecation",
          "-unchecked",
          "-feature",
          "-target:jvm-1.8"
        ),
    javacOptions ++= Seq("-source", "1.8", "-target", "1.8"),
    libraryDependencies ++= Seq(
      ("org.scalanlp" %% "breeze" % "2.1.0"),
      ("org.scalanlp" %% "breeze-natives" % "2.1.0"),
      ("ch.unibas.cs.gravis" %% "scaltair" % "0.1-SNAPSHOT"),
      "org.scalameta" %% "munit" % "0.7.29" % Test
    )
  )
