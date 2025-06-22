import { useState } from "react";
import "./App.css";
import { GoogleGeminiEffect } from "./components/google-gemini.jsx";
import { useScroll, useTransform } from "motion/react";
import { motion } from "motion/react";
import React from "react";
import { Button } from "./components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./components/ui/card";
import { Avatar, AvatarFallback, AvatarImage } from "./components/ui/avatar";
import { Progress } from "./components/ui/progress";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "./components/ui/dialog";
import { Code, Zap, Shield, User, Settings, Download, Upload, Play } from "lucide-react";
import FloatingStars from "./components/sparkles";

function App({ onNavigateToWork, onNavigateToUpload }) {
  const containerRef = React.useRef(null);
  const featuresRef = React.useRef(null);

  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start end", "end start"],
  });

  const { scrollYProgress: featuresScrollProgress } = useScroll({
    target: featuresRef,
    offset: ["start end", "end start"],
  });

  return (
    <div ref={containerRef} className="min-h-screen bg-gray-900 text-white pb-100">
      {/* Floating Stars */}
      <FloatingStars count={10} />
      
      {/* Header */}
      <motion.header 
        className="py-4 shadow-lg bg-gray-800/50 backdrop-blur-sm border-b border-gray-700"
        initial={{ y: -100, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.8, ease: "easeOut" }}
      >
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            {/* Logo and App Name */}
            <motion.div 
              className="flex items-center space-x-3"
              initial={{ x: -50, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              transition={{ duration: 0.8, delay: 0.2, ease: "easeOut" }}
            >
              <motion.div 
                className="w-10 h-10 bg-gradient-to-br from-blue-400 to-blue-600 rounded-lg flex items-center justify-center"
                whileHover={{ scale: 1.1, rotate: 5 }}
                transition={{ duration: 0.2 }}
              >
                <span className="text-white font-bold text-lg">P</span>
              </motion.div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
                  Py2STM
                </h1>
                <p className="text-xs text-gray-400">Python to STM32</p>
              </div>
            </motion.div>

            {/* Navigation and User Menu */}
            <motion.div 
              className="flex items-center space-x-4"
              initial={{ x: 50, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              transition={{ duration: 0.8, delay: 0.4, ease: "easeOut" }}
            >
              <nav className="hidden md:flex items-center space-x-6">
                {onNavigateToWork && (
                  <Button
                    variant="ghost"
                    onClick={onNavigateToWork}
                    className="text-gray-300 hover:text-blue-400"
                  >
                    Home
                  </Button>
                )}
                {onNavigateToUpload && (
                  <Button
                    variant="ghost"
                    onClick={onNavigateToUpload}
                    className="text-gray-300 hover:text-blue-400"
                  >
                    MNIST Demo
                  </Button>
                )}
              </nav>
              <Dialog>
                <DialogTrigger asChild>
                  <Button variant="gradient" className="font-medium">
                    <Upload className="w-4 h-4 mr-2" />
                    New Project
                  </Button>
                </DialogTrigger>
                <DialogContent className="bg-gray-800 border-gray-700">
                  <DialogHeader>
                    <DialogTitle className="text-white">Create New Project</DialogTitle>
                    <DialogDescription className="text-gray-400">
                      Start a new Python to STM32 conversion project
                    </DialogDescription>
                  </DialogHeader>
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <label className="text-sm font-medium text-gray-300">Project Name</label>
                      <input 
                        type="text" 
                        className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
                        placeholder="Enter project name"
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium text-gray-300">Python File</label>
                      <input 
                        type="file" 
                        accept=".py"
                        className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                      />
                    </div>
                    <Button variant="gradient" className="w-full">
                      <Play className="w-4 h-4 mr-2" />
                      Create Project
                    </Button>
                  </div>
                </DialogContent>
              </Dialog>
              <Avatar>
                <AvatarImage src="/avatars/user.png" alt="User" />
                <AvatarFallback className="bg-blue-600 text-white">
                  <User className="w-4 h-4" />
                </AvatarFallback>
              </Avatar>
            </motion.div>
          </div>
        </div>
      </motion.header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <motion.div 
          className="bg-gray-800 rounded-lg p-6 border border-gray-700"
          initial={{ y: 50, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.6, ease: "easeOut" }}
        >
          <motion.h2 
            className="text-xl font-semibold text-gray-100 mb-4"
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.6, delay: 0.8, ease: "easeOut" }}
          >
            Welcome to Py2STM
          </motion.h2>
          <motion.p 
            className="text-gray-400 leading-relaxed mb-6"
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.6, delay: 1, ease: "easeOut" }}
          >
            Py2STM is your comprehensive platform for converting Python code to
            STM32 microcontroller applications. Build, compile, and deploy your
            Python projects directly to STM32 devices with ease.
          </motion.p>

          {/* Stats Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
            <Card className="bg-gray-700/50 border-gray-600">
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-2xl font-bold text-blue-400">24</p>
                    <p className="text-sm text-gray-400">Active Projects</p>
                  </div>
                  <div className="w-12 h-12 bg-blue-500/20 rounded-lg flex items-center justify-center">
                    <Code className="w-6 h-6 text-blue-400" />
                  </div>
                </div>
              </CardContent>
            </Card>
            <Card className="bg-gray-700/50 border-gray-600">
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-2xl font-bold text-green-400">156</p>
                    <p className="text-sm text-gray-400">Conversions</p>
                  </div>
                  <div className="w-12 h-12 bg-green-500/20 rounded-lg flex items-center justify-center">
                    <Zap className="w-6 h-6 text-green-400" />
                  </div>
                </div>
              </CardContent>
            </Card>
            <Card className="bg-gray-700/50 border-gray-600">
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-2xl font-bold text-purple-400">98%</p>
                    <p className="text-sm text-gray-400">Success Rate</p>
                  </div>
                  <div className="w-12 h-12 bg-purple-500/20 rounded-lg flex items-center justify-center">
                    <Shield className="w-6 h-6 text-purple-400" />
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Feature Cards */}
          <div ref={featuresRef} className="grid md:grid-cols-3 gap-6 mt-8">
            <motion.div
              initial={{ y: 100, opacity: 0, scale: 0.8 }}
              whileInView={{ y: 0, opacity: 1, scale: 1 }}
              transition={{ duration: 0.8, delay: 0.2, ease: "easeOut" }}
              viewport={{ once: true }}
            >
              <Card className="bg-gray-700/50 border-gray-600 hover:border-blue-400/50 transition-colors duration-200 h-full">
                <CardHeader className="pb-4">
                  <div className="w-12 h-12 bg-blue-500/20 rounded-lg flex items-center justify-center mb-4">
                    <Code className="w-6 h-6 text-blue-400" />
                  </div>
                  <CardTitle className="text-lg text-gray-100">Code Conversion</CardTitle>
                </CardHeader>
                <CardContent>
                  <CardDescription className="text-gray-400">
                    Convert Python code to optimized C/C++ for STM32 microcontrollers with advanced compilation techniques.
                  </CardDescription>
                  <div className="mt-4">
                    <Progress value={85} className="h-2" />
                    <p className="text-xs text-gray-500 mt-1">85% optimization rate</p>
                  </div>
                </CardContent>
              </Card>
            </motion.div>

            <motion.div
              initial={{ y: 100, opacity: 0, scale: 0.8 }}
              whileInView={{ y: 0, opacity: 1, scale: 1 }}
              transition={{ duration: 0.8, delay: 0.4, ease: "easeOut" }}
              viewport={{ once: true }}
            >
              <Card className="bg-gray-700/50 border-gray-600 hover:border-green-400/50 transition-colors duration-200 h-full">
                <CardHeader className="pb-4">
                  <div className="w-12 h-12 bg-green-500/20 rounded-lg flex items-center justify-center mb-4">
                    <Zap className="w-6 h-6 text-green-400" />
                  </div>
                  <CardTitle className="text-lg text-gray-100">Real-time Debugging</CardTitle>
                </CardHeader>
                <CardContent>
                  <CardDescription className="text-gray-400">
                    Debug your applications in real-time with advanced monitoring tools and live code analysis.
                  </CardDescription>
                  <div className="mt-4">
                    <Progress value={92} className="h-2" />
                    <p className="text-xs text-gray-500 mt-1">92% accuracy</p>
                  </div>
                </CardContent>
              </Card>
            </motion.div>

            <motion.div
              initial={{ y: 100, opacity: 0, scale: 0.8 }}
              whileInView={{ y: 0, opacity: 1, scale: 1 }}
              transition={{ duration: 0.8, delay: 0.6, ease: "easeOut" }}
              viewport={{ once: true }}
            >
              <Card className="bg-gray-700/50 border-gray-600 hover:border-purple-400/50 transition-colors duration-200 h-full">
                <CardHeader className="pb-4">
                  <div className="w-12 h-12 bg-purple-500/20 rounded-lg flex items-center justify-center mb-4">
                    <Download className="w-6 h-6 text-purple-400" />
                  </div>
                  <CardTitle className="text-lg text-gray-100">Easy Deployment</CardTitle>
                </CardHeader>
                <CardContent>
                  <CardDescription className="text-gray-400">
                    Deploy your applications directly to STM32 devices with one-click deployment and automated testing.
                  </CardDescription>
                  <div className="mt-4">
                    <Progress value={78} className="h-2" />
                    <p className="text-xs text-gray-500 mt-1">78% faster deployment</p>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          </div>

          {/* Recent Activity */}
          <motion.div 
            className="mt-8"
            initial={{ y: 50, opacity: 0 }}
            whileInView={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.8, delay: 0.8, ease: "easeOut" }}
            viewport={{ once: true }}
          >
            <h3 className="text-lg font-semibold text-gray-100 mb-4">Recent Activity</h3>
            <div className="space-y-3">
              {[
                { name: "MNIST Model", status: "Completed", time: "2 hours ago", icon: "🎯" },
                { name: "Image Recognition", status: "In Progress", time: "4 hours ago", icon: "🖼️" },
                { name: "Audio Processing", status: "Queued", time: "6 hours ago", icon: "🎵" }
              ].map((activity, index) => (
                <motion.div
                  key={index}
                  className="flex items-center justify-between p-3 bg-gray-700/30 rounded-lg border border-gray-600/50"
                  initial={{ x: -50, opacity: 0 }}
                  whileInView={{ x: 0, opacity: 1 }}
                  transition={{ duration: 0.6, delay: index * 0.1, ease: "easeOut" }}
                  viewport={{ once: true }}
                >
                  <div className="flex items-center space-x-3">
                    <span className="text-2xl">{activity.icon}</span>
                    <div>
                      <p className="text-gray-200 font-medium">{activity.name}</p>
                      <p className="text-sm text-gray-400">{activity.time}</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                      activity.status === 'Completed' ? 'bg-green-500/20 text-green-400' :
                      activity.status === 'In Progress' ? 'bg-blue-500/20 text-blue-400' :
                      'bg-yellow-500/20 text-yellow-400'
                    }`}>
                      {activity.status}
                    </span>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </motion.div>
      </main>
    </div>
  );
}

export default App;
