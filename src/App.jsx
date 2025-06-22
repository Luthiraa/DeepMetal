import { useState } from "react";
import "./App.css";
import { GoogleGeminiEffect } from "./components/google-gemini.jsx";
import { useScroll, useTransform } from "motion/react";
import React from "react";

function App() {
  return (
    <div className="min-h-screen bg-gray-900 text-white pb-100">
      {/* Header */}
      <header className="py-4 shadow-lg">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            {/* Logo and App Name */}
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-400 to-blue-600 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-lg ">P</span>
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
                  Py2STM
                </h1>
                <p className="text-xs text-gray-400">Python to STM32</p>
              </div>
            </div>

            {/* User Menu */}
            <div className="flex items-center space-x-4">
              <button className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-lg transition-colors duration-200 font-medium">
                New Project
              </button>
              <div className="w-8 h-8 bg-gray-700 rounded-full flex items-center justify-center">
                <span className="text-gray-300 text-sm font-medium">U</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h2 className="text-xl font-semibold text-gray-100 mb-4">
            Welcome to Py2STM
          </h2>
          <p className="text-gray-400 leading-relaxed">
            Py2STM is your comprehensive platform for converting Python code to
            STM32 microcontroller applications. Build, compile, and deploy your
            Python projects directly to STM32 devices with ease.
          </p>

          {/* Feature Cards */}
          <div className="grid md:grid-cols-3 gap-6 mt-8">
            <div className="bg-gray-700 rounded-lg p-4 border border-gray-600 hover:border-blue-400/50 transition-colors duration-200">
              <div className="w-12 h-12 bg-blue-500 rounded-lg flex items-center justify-center mb-4">
                <svg
                  className="w-6 h-6 text-white"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4"
                  />
                </svg>
              </div>
              <h3 className="text-lg font-medium text-gray-100 mb-2">
                Code Conversion
              </h3>
              <p className="text-gray-400 text-sm">
                Convert Python code to optimized C/C++ for STM32
                microcontrollers
              </p>
            </div>

            <div className="bg-gray-700 rounded-lg p-4 border border-gray-600 hover:border-blue-400/50 transition-colors duration-200">
              <div className="w-12 h-12 bg-blue-500 rounded-lg flex items-center justify-center mb-4">
                <svg
                  className="w-6 h-6 text-white"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
              </div>
              <h3 className="text-lg font-medium text-gray-100 mb-2">
                Real-time Debugging
              </h3>
              <p className="text-gray-400 text-sm">
                Debug your applications in real-time with advanced monitoring
                tools
              </p>
            </div>

            <div className="bg-gray-700 rounded-lg p-4 border border-gray-600 hover:border-blue-400/50 transition-colors duration-200">
              <div className="w-12 h-12 bg-blue-500 rounded-lg flex items-center justify-center mb-4">
                <svg
                  className="w-6 h-6 text-white"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"
                  />
                </svg>
              </div>
              <h3 className="text-lg font-medium text-gray-100 mb-2">
                Easy Deployment
              </h3>
              <p className="text-gray-400 text-sm">
                Deploy your applications directly to STM32 devices with one
                click
              </p>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
