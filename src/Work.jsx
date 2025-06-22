import { GoogleGeminiEffect } from "./components/google-gemini";
import { useScroll, useTransform, motion } from "motion/react";
import React from "react";
import FloatingStars from "./components/sparkles";

export default function Work({ onNavigateToUpload, onNavigateToApp }) {
  const ref = React.useRef(null);
  const heroRef = React.useRef(null);
  const featuresRef = React.useRef(null);
  const statsRef = React.useRef(null);
  
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ["start center", "end start"],
  });
  
  const { scrollYProgress: heroScrollProgress } = useScroll({
    target: heroRef,
    offset: ["start end", "end start"],
  });

  const { scrollYProgress: featuresScrollProgress } = useScroll({
    target: featuresRef,
    offset: ["start end", "end start"],
  });

  const { scrollYProgress: statsScrollProgress } = useScroll({
    target: statsRef,
    offset: ["start end", "end start"],
  });

  const pathLengthFirst = useTransform(scrollYProgress, [0, 0.7], [0.2, 1.2]);
  const pathLengthSecond = useTransform(scrollYProgress, [0, 0.7], [0.15, 1.2]);
  const pathLengthThird = useTransform(scrollYProgress, [0, 0.7], [0.1, 1.2]);
  const pathLengthFourth = useTransform(scrollYProgress, [0, 0.7], [0.05, 1.2]);
  const pathLengthFifth = useTransform(scrollYProgress, [0, 0.7], [0, 1.2]);
  const iconOpacity = useTransform(scrollYProgress, [0, 0.2], [1, 0]);
  const otherOpacity = useTransform(scrollYProgress, [0, 0.9], [0, 1]);

  return (
    <div className="w-full relative flex flex-col min-h-screen bg-[linear-gradient(180deg,_#101827_0%,_#0F1C4D_28%,_#1A2B6D_57%,_#22397E_84%,_#2A4690_100%)] text-white max-w-screen overflow-x-hidden">
      {/* Floating Stars */}
      <FloatingStars count={12} />
      
      {/* Header */}
      <div className="absolute top-1/2 -left-20 w-full h-full  -translate-y-1/2 bg-[linear-gradient(140deg,_#000000_0%,_#00000000_30%)] pointer-events-none -z-0"></div>
      <div className="absolute top-1/2  w-full h-full  -translate-y-1/2 bg-[linear-gradient(210deg,_#00000000_50%,_#0000005c_75%,_#0000005c_85%,_#00000000_100%)] pointer-events-none -z-0"></div>
      <motion.header 
        className="fixed top-0 left-0 w-full z-50 bg-[#0a101f]/50 backdrop-blur-sm py-8 px-8 lg:px-16 flex justify-between items-center"
        initial={{ y: -100, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.8, ease: "easeOut" }}
      >
        <motion.div 
          className="flex-row items-center justify-center flex space-x-4"
          initial={{ x: -50, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.2, ease: "easeOut" }}
        >
          <div className="text-4xl font-bold">Py2STM</div>
          <img src="logo.svg" alt="logo" className="w-10 h-10" />
        </motion.div>

        <motion.nav 
          className="hidden md:flex items-center space-x-8 font-bold text-xl"
          initial={{ x: 50, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.4, ease: "easeOut" }}
        >
          <a href="#" className="hover:text-blue-400 transition-colors">
            Home
          </a>
          {/* <a href="#" className="hover:text-blue-400 transition-colors">
            Our Mission
          </a>
          <a href="#" className="hover:text-blue-400 transition-colors">
            About
          </a> */}
          {onNavigateToApp && (
            <button
              onClick={onNavigateToApp}
              className="hover:text-blue-400 transition-colors"
            >
              Dashboard
            </button>
          )}
          {onNavigateToUpload && (
            <motion.button
              onClick={onNavigateToUpload}
              className="bg-blue-500 hover:bg-blue-600 text-white px-6 py-2 rounded-lg transition-colors duration-200 font-medium"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              Try MNIST
            </motion.button>
          )}
        </motion.nav>
      </motion.header>

      <main className="pt-24">
        {/* Hero Section */}
        <section ref={heroRef} className="h-[calc(100vh-6rem)] flex items-center justify-center px-8 lg:px-16">
          <div className="flex flex-col md:flex-row items-center w-full">
            {/* Left Side: 3D Model Placeholder */}
            <motion.div 
              className="w-full md:w-1/2 flex justify-center mb-10 md:mb-0"
              initial={{ x: -100, opacity: 0 }}
              whileInView={{ x: 0, opacity: 1 }}
              transition={{ duration: 1, ease: "easeOut" }}
              viewport={{ once: true }}
            >
              <div className="w-96 h-96 relative">
                {/* Placeholder for 3D Python logos */}
                <motion.div 
                  className="w-64 h-64 bg-gray-700/30 rounded-3xl absolute top-10 left-10 transform rotate-[-15deg] shadow-2xl flex items-center justify-center text-6xl backdrop-blur-sm border border-gray-600/50"
                  style={{
                    x: useTransform(heroScrollProgress, [0, 0.5], [0, 0]),
                    y: useTransform(heroScrollProgress, [0, 0.5], [0, -50]),
                    rotate: useTransform(heroScrollProgress, [0, 0.5], [-15, -25]),
                    opacity: useTransform(heroScrollProgress, [0, 0.3], [1, 0.8]),
                  }}
                  whileHover={{ scale: 1.05, rotate: -20 }}
                  transition={{ duration: 0.3 }}
                >
                  🐍
                </motion.div>
                <motion.div 
                  className="w-64 h-72 bg-gray-600/40 rounded-3xl absolute top-32 left-32 transform shadow-2xl flex items-center justify-center text-6xl backdrop-blur-sm border border-gray-500/50"
                  style={{
                    x: useTransform(heroScrollProgress, [0, 0.5], [0, 100]),
                    y: useTransform(heroScrollProgress, [0, 0.5], [0, -50]),
                    rotate: useTransform(heroScrollProgress, [0, 0.5], [10, 25]),
                    opacity: useTransform(heroScrollProgress, [0, 0.3], [1, 0.8]),
                  }}
                  whileHover={{ scale: 1.05, rotate: 15 }}
                  transition={{ duration: 0.3 }}
                >
                  <img className="w-32 h-32" src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/C_Programming_Language.svg/926px-C_Programming_Language.svg.png" alt="python" />
                </motion.div>
              </div>
            </motion.div>

            {/* Right Side: Text Content */}
            <motion.div 
              className="w-full md:w-1/2 text-center md:text-left"
              initial={{ x: 100, opacity: 0 }}
              whileInView={{ x: 0, opacity: 1 }}
              transition={{ duration: 1, delay: 0.3, ease: "easeOut" }}
              viewport={{ once: true }}
            >
              <motion.h1 
                className="text-4xl lg:text-5xl font-bold leading-tight"
                initial={{ y: 50, opacity: 0 }}
                whileInView={{ y: 0, opacity: 1 }}
                transition={{ duration: 0.8, delay: 0.5, ease: "easeOut" }}
                viewport={{ once: true }}
              >
                Efficiency of Higher Level <br /> Languages at a Lower Cost
              </motion.h1>
              <motion.p 
                className="mt-6 text-base lg:text-lg text-gray-300 max-w-md mx-auto md:mx-0"
                initial={{ y: 50, opacity: 0 }}
                whileInView={{ y: 0, opacity: 1 }}
                transition={{ duration: 0.8, delay: 0.7, ease: "easeOut" }}
                viewport={{ once: true }}
              >
                We are changing the game with the creation of new ML and AI
                integrated products, by creating a cheaper method to implement
                your project.
              </motion.p>
              {onNavigateToUpload && (
                <motion.button
                  onClick={onNavigateToUpload}
                  className="mt-8 bg-blue-500 hover:bg-blue-600 text-white px-8 py-3 rounded-lg transition-colors duration-200 font-medium text-lg"
                  initial={{ y: 50, opacity: 0 }}
                  whileInView={{ y: 0, opacity: 1 }}
                  transition={{ duration: 0.8, delay: 0.9, ease: "easeOut" }}
                  viewport={{ once: true }}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  Try MNIST Recognition →
                </motion.button>
              )}
            </motion.div>
          </div>
        </section>

        {/* Features Section */}
        <section ref={featuresRef} className="py-24 px-8 lg:px-16">
          <motion.div 
            className="text-center mb-16"
            initial={{ y: 100, opacity: 0 }}
            whileInView={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.8, ease: "easeOut" }}
            viewport={{ once: true }}
          >
            <h2 className="text-4xl font-bold text-white mb-6">Key Features</h2>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto">
              Transform your Python machine learning projects into optimized STM32 applications with our advanced conversion technology.
            </p>
          </motion.div>
          
          <div className="flex flex-wrap justify-center gap-8">
            {/* Feature 1: Code Conversion */}
            <motion.div 
              className="w-80 h-80 bg-gray-800/20 rounded-2xl backdrop-blur-sm border border-gray-700/50 p-8 flex flex-col items-center text-center"
              initial={{ y: 100, opacity: 0, scale: 0.8 }}
              whileInView={{ y: 0, opacity: 1, scale: 1 }}
              transition={{ duration: 0.8, delay: 0.2, ease: "easeOut" }}
              viewport={{ once: true }}
              whileHover={{ y: -10, scale: 1.02 }}
            >
              <motion.div 
                className="w-16 h-16 bg-blue-500/20 rounded-full flex items-center justify-center mb-6"
                whileHover={{ rotate: 360 }}
                transition={{ duration: 0.6 }}
              >
                <svg className="w-8 h-8 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
                </svg>
              </motion.div>
              <h3 className="text-xl font-bold text-white mb-4">Code Conversion</h3>
              <p className="text-gray-300 text-sm leading-relaxed">
                Seamlessly convert Python ML libraries to optimized C code for STM32 microcontrollers. 
                Support for popular frameworks like TensorFlow, PyTorch, and scikit-learn.
              </p>
            </motion.div>

            {/* Feature 2: Streamlined Implementation */}
            <motion.div 
              className="w-80 h-80 bg-gray-800/20 rounded-2xl backdrop-blur-sm border border-gray-700/50 p-8 flex flex-col items-center text-center"
              initial={{ y: 100, opacity: 0, scale: 0.8 }}
              whileInView={{ y: 0, opacity: 1, scale: 1 }}
              transition={{ duration: 0.8, delay: 0.4, ease: "easeOut" }}
              viewport={{ once: true }}
              whileHover={{ y: -10, scale: 1.02 }}
            >
              <motion.div 
                className="w-16 h-16 bg-green-500/20 rounded-full flex items-center justify-center mb-6"
                whileHover={{ rotate: 360 }}
                transition={{ duration: 0.6 }}
              >
                <svg className="w-8 h-8 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" />
                </svg>
              </motion.div>
              <h3 className="text-xl font-bold text-white mb-4">Streamlined Process</h3>
              <p className="text-gray-300 text-sm leading-relaxed">
                Simplify your development workflow with automated code generation, 
                memory optimization, and one-click deployment to STM32 devices.
              </p>
            </motion.div>

            {/* Feature 3: Increased Performance */}
            <motion.div 
              className="w-80 h-80 bg-gray-800/20 rounded-2xl backdrop-blur-sm border border-gray-700/50 p-8 flex flex-col items-center text-center"
              initial={{ y: 100, opacity: 0, scale: 0.8 }}
              whileInView={{ y: 0, opacity: 1, scale: 1 }}
              transition={{ duration: 0.8, delay: 0.6, ease: "easeOut" }}
              viewport={{ once: true }}
              whileHover={{ y: -10, scale: 1.02 }}
            >
              <motion.div 
                className="w-16 h-16 bg-purple-500/20 rounded-full flex items-center justify-center mb-6"
                whileHover={{ rotate: 360 }}
                transition={{ duration: 0.6 }}
              >
                <svg className="w-8 h-8 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </motion.div>
              <h3 className="text-xl font-bold text-white mb-4">Increased Performance</h3>
              <p className="text-gray-300 text-sm leading-relaxed">
                Achieve up to 10x faster inference times with our optimized C implementations. 
                Reduced memory footprint and improved power efficiency for edge devices.
              </p>
            </motion.div>
          </div>
        </section>

        {/* Stats Section */}
        <section ref={statsRef} className="py-24 px-8 lg:px-16 flex flex-col md:flex-row items-center gap-16">
          <motion.div 
            className="w-full md:w-1/2 text-center"
            initial={{ x: -100, opacity: 0 }}
            whileInView={{ x: 0, opacity: 1 }}
            transition={{ duration: 0.8, ease: "easeOut" }}
            viewport={{ once: true }}
          >
            <p className="text-3xl text-gray-300 mx-auto w-[70%]">
              Some stats and further description on the benefits of using our
              product.
            </p>
          </motion.div>
          <motion.div 
            className="w-full md:w-1/2 text-center"
            initial={{ x: 100, opacity: 0 }}
            whileInView={{ x: 0, opacity: 1 }}
            transition={{ duration: 0.8, delay: 0.3, ease: "easeOut" }}
            viewport={{ once: true }}
          >
            <motion.p 
              className="text-8xl lg:text-9xl font-bold"
              initial={{ scale: 0.5, opacity: 0 }}
              whileInView={{ scale: 1, opacity: 1 }}
              transition={{ duration: 1, delay: 0.5, ease: "easeOut" }}
              viewport={{ once: true }}
            >
              3x
            </motion.p>
            <motion.p 
              className="text-4xl lg:text-5xl mt-4"
              initial={{ y: 50, opacity: 0 }}
              whileInView={{ y: 0, opacity: 1 }}
              transition={{ duration: 0.8, delay: 0.7, ease: "easeOut" }}
              viewport={{ once: true }}
            >
              Cost Savings
            </motion.p>
          </motion.div>
        </section>

        {/* Funnel Section */}
        <div ref={ref} className="relative">
          <GoogleGeminiEffect
            pathLengths={[
              pathLengthFirst,
              pathLengthSecond,
              pathLengthThird,
              pathLengthFourth,
              pathLengthFifth,
            ]}
            iconOpacity={iconOpacity}
            otherOpacity={otherOpacity}
            className=""
          />
        </div>
      </main>
    </div>
  );
}
