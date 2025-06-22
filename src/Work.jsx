import { GoogleGeminiEffect } from "./components/google-gemini";
import { useScroll, useTransform, motion } from "motion/react";
import React from "react";

export default function Work() {
  const ref = React.useRef(null);
  const heroRef = React.useRef(null);
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ["start center", "end start"],
  });
  
  const { scrollYProgress: heroScrollProgress } = useScroll({
    target: heroRef,
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
      {/* Header */}
      <div className="absolute top-1/2 -left-20 w-full h-full  -translate-y-1/2 bg-[linear-gradient(140deg,_#000000_0%,_#00000000_30%)] pointer-events-none -z-0"></div>
      <div className="absolute top-1/2  w-full h-full  -translate-y-1/2 bg-[linear-gradient(210deg,_#00000000_50%,_#0000005c_75%,_#0000005c_85%,_#00000000_100%)] pointer-events-none -z-0"></div>
      <header className=" fixed top-0 left-0 w-full z-50  bg-[#0a101f]/50 backdrop-blur-sm py-8 px-8 lg:px-16 flex justify-between items-center">
        <div className="flex-row items-center justify-center flex  space-x-4">
          <div className="text-4xl font-bold">Py2STM</div>
          <img src="logo.svg" alt="logo" className="w-10 h-10" />
        </div>

        <nav className="hidden md:flex items-center space-x-8 font-bold text-xl">
          <a href="#" className="hover:text-blue-400 transition-colors">
            Home
          </a>
          <a href="#" className="hover:text-blue-400 transition-colors">
            Our Mission
          </a>
          <a href="#" className="hover:text-blue-400 transition-colors">
            About
          </a>
        </nav>
      </header>

      <main className="pt-24">
        {/* Hero Section */}
        <section ref={heroRef} className="h-[calc(100vh-6rem)] flex items-center justify-center px-8 lg:px-16">
          <div className="flex flex-col md:flex-row items-center w-full">
            {/* Left Side: 3D Model Placeholder */}
            <div className="w-full md:w-1/2 flex justify-center mb-10 md:mb-0">
              <div className="w-96 h-96 relative">
                {/* Placeholder for 3D Python logos */}
                <motion.div 
                  className="w-64 h-64 bg-gray-700/30 rounded-3xl absolute top-10 left-10 transform rotate-[-15deg] shadow-2xl flex items-center justify-center text-6xl backdrop-blur-sm border border-gray-600/50"
                  style={{
                    x: useTransform(heroScrollProgress, [0, 0.5], [0, -200]),
                    y: useTransform(heroScrollProgress, [0, 0.5], [0, -50]),
                    rotate: useTransform(heroScrollProgress, [0, 0.5], [-15, -25]),
                    opacity: useTransform(heroScrollProgress, [0, 0.3], [1, 0.8]),
                  }}
                >
                  🐍
                </motion.div>
                <motion.div 
                  className="w-64 h-64 bg-gray-600/40 rounded-3xl absolute top-32 left-32 transform  shadow-2xl flex items-center justify-center text-6xl backdrop-blur-sm border border-gray-500/50"
                  style={{
                    x: useTransform(heroScrollProgress, [0, 0.5], [0, 100]),
                    y: useTransform(heroScrollProgress, [0, 0.5], [0, -50]),
                    rotate: useTransform(heroScrollProgress, [0, 0.5], [10, 25]),
                    opacity: useTransform(heroScrollProgress, [0, 0.3], [1, 0.8]),
                  }}
                >
                  🐍
                </motion.div>
              </div>
            </div>

            {/* Right Side: Text Content */}
            <div className="w-full md:w-1/2 text-center md:text-left">
              <h1 className="text-4xl lg:text-5xl font-bold leading-tight">
                Efficiency of Higher Level <br /> Languages at a Lower Cost
              </h1>
              <p className="mt-6 text-base lg:text-lg text-gray-300 max-w-md mx-auto md:mx-0">
                We are changing the game with the creation of new ML and AI
                integrated products, by creating a cheaper method to implement
                your project.
              </p>
            </div>
          </div>
        </section>

        {/* Features Section */}
        <section className="py-24 px-8 lg:px-16">
          <div className="flex flex-wrap justify-center gap-8">
            <div className="w-80 h-64 bg-gray-800/20 rounded-2xl backdrop-blur-sm border border-gray-700/50"></div>
            <div className="w-80 h-64 bg-gray-800/20 rounded-2xl backdrop-blur-sm border border-gray-700/50"></div>
            <div className="w-80 h-64 bg-gray-800/20 rounded-2xl backdrop-blur-sm border border-gray-700/50"></div>
          </div>
        </section>

        {/* Stats Section */}
        <section className="py-24 px-8 lg:px-16 flex flex-col md:flex-row items-center gap-16">
          <div className="w-full md:w-1/2 text-center">
            <p className="text-3xl text-gray-300 mx-auto w-[70%]">
              Some stats and further description on the benefits of using our
              product.
            </p>
          </div>
          <div className="w-full md:w-1/2 text-center">
            <p className="text-8xl lg:text-9xl font-bold">3x</p>
            <p className="text-4xl lg:text-5xl mt-4">Cost Savings</p>
          </div>
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
