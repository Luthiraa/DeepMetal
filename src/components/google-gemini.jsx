"use client";
import { cn } from "../../lib/utils";
import { motion } from "motion/react";
import React from "react";

const transition = {
  duration: 0,
  ease: "linear",
};

export const GoogleGeminiEffect = ({
  pathLengths,
  title,
  description,
  className,
  iconOpacity = 1,
  otherOpacity = 0,
}) => {
  const iconPositions = [
    [100, "https://imgs.search.brave.com/sv3Rby54dx8w72WilWm79wZ6Uv65nh-ZCVbYsej4-g0/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9pbWFn/ZXMuc2Vla2xvZ28u/Y29tL2xvZ28tcG5n/LzQzLzIvdGVuc29y/Zmxvdy1sb2dvLXBu/Z19zZWVrbG9nby00/MzUxMjQucG5n"], 
    [200, "https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Scikit_learn_logo_small.svg/1200px-Scikit_learn_logo_small.svg.png"], 
    [300, "https://static-00.iconduck.com/assets.00/pytorch-icon-1694x2048-jgwjy3ne.png"], 
    [400, "https://studyopedia.com/wp-content/uploads/2023/07/scipy.png"], 
    [500, "https://upload.wikimedia.org/wikipedia/commons/thumb/0/01/Created_with_Matplotlib-logo.svg/2048px-Created_with_Matplotlib-logo.svg.png"], 
    [940, "https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/Theano_logo.svg/1280px-Theano_logo.svg.png"], 
    [1040, "https://cdn.worldvectorlogo.com/logos/seaborn-1.svg"], 
    [1140, "https://img.icons8.com/color/512/pandas.png"], 
    [1240, "https://upload.wikimedia.org/wikipedia/commons/thumb/3/31/NumPy_logo_2020.svg/2560px-NumPy_logo_2020.svg.png"], 
    [1340, "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQOHx6oZAsXTmwSqwm8Sehgd8eT6gThlzACSA&s"]
  ];
  return (
    <div className={cn("relative", className)}>
      {/* <p
        className="my-auto text-lg md:text-7xl font-normal pb-4 text-center bg-clip-text text-transparent bg-gradient-to-b from-neutral-100 to-neutral-300">
        {title || `Build with Py2STM`}
      </p>
      <p
        className="text-xs md:text-xl font-normal text-center text-neutral-400 mt-4 max-w-lg mx-auto">
        {description ||
          `Convert Python code to STM32 microcontroller applications with ease!`}
      </p> */}
      <div
        className="w-full h-[1900px] -mt-40 flex items-center justify-center bg-transparent relative">
        {/* <button
          className="font-bold bg-blue-500 hover:bg-blue-600 rounded-full md:px-4 md:py-2 px-2 py-1 md:mt-24 mt-8 z-30 md:text-base text-white text-xs  w-fit mx-auto transition-colors duration-200">
          Get Started
        </button> */}
      </div>
      <svg
        width="1440"
        height="1900"
        viewBox="0 0 1440 1200"
        xmlns="http://www.w3.org/2000/svg"
        className="absolute top-0 left-0 w-full h-full overflow-visible">
        
        {/* Icons at the top of each funnel line */}
        {iconPositions.map((x, index) => (
          <motion.image
            key={index}
            href={x[1]}
            x={x[0] - 40}
            y={-80 }
            
            width="100"
            height="80"
            style={{ opacity: iconOpacity }}
            className="-z-5"
          />
        ))}
        
        {/* Image at the end of the funnel where all lines converge */}
        <motion.image
          href="https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/C_Programming_Language.svg/926px-C_Programming_Language.svg.png"
          x={720 - 100}
          y={1250}
          width="200"
          height="200"
          style={{ opacity: otherOpacity }}
          className="-z-5"
        />
        
        <motion.path
          d="M100 0C100 150 200 250 300 350C400 450 500 550 600 650C650 750 700 850 720 950C720 1050 720 1150 720 1200"
          stroke="#3B82F6"
          strokeWidth="2"
          fill="none"
          initial={{
            pathLength: 0,
          }}
          style={{
            pathLength: pathLengths[0],
          }}
          transition={transition}
        />
        <motion.path
          d="M200 0C200 120 300 220 400 320C500 420 550 520 600 620C650 720 680 820 720 920C720 1020 720 1120 720 1200"
          stroke="#60A5FA"
          strokeWidth="2"
          fill="none"
          initial={{
            pathLength: 0,
          }}
          style={{
            pathLength: pathLengths[1],
          }}
          transition={transition}
        />
        <motion.path
          d="M300 0C300 100 400 200 500 300C550 400 600 500 650 600C680 700 700 800 720 900C720 1000 720 1100 720 1200"
          stroke="#93C5FD"
          strokeWidth="2"
          fill="none"
          initial={{
            pathLength: 0,
          }}
          style={{
            pathLength: pathLengths[2],
          }}
          transition={transition}
        />
        <motion.path
          d="M400 0C400 80 500 180 550 280C600 380 650 480 680 580C700 680 720 780 720 880C720 980 720 1080 720 1200"
          stroke="#BFDBFE"
          strokeWidth="2"
          fill="none"
          initial={{
            pathLength: 0,
          }}
          style={{
            pathLength: pathLengths[3],
          }}
          transition={transition}
        />
        <motion.path
          d="M500 0C500 60 550 160 600 260C650 360 680 460 700 560C720 660 720 760 720 860C720 960 720 1060 720 1200"
          stroke="#DBEAFE"
          strokeWidth="2"
          fill="none"
          initial={{
            pathLength: 0,
          }}
          style={{
            pathLength: pathLengths[4],
          }}
          transition={transition}
        />
        <motion.path
          d="M1340 0C1340 150 1240 250 1140 350C1040 450 940 550 840 650C790 750 740 850 720 950C720 1050 720 1150 720 1200"
          stroke="#3B82F6"
          strokeWidth="2"
          fill="none"
          initial={{
            pathLength: 0,
          }}
          style={{
            pathLength: pathLengths[0],
          }}
          transition={transition}
        />
        <motion.path
          d="M1240 0C1240 120 1140 220 1040 320C940 420 890 520 840 620C790 720 760 820 720 920C720 1020 720 1120 720 1200"
          stroke="#60A5FA"
          strokeWidth="2"
          fill="none"
          initial={{
            pathLength: 0,
          }}
          style={{
            pathLength: pathLengths[1],
          }}
          transition={transition}
        />
        <motion.path
          d="M1140 0C1140 100 1040 200 940 300C890 400 840 500 790 600C760 700 740 800 720 900C720 1000 720 1100 720 1200"
          stroke="#93C5FD"
          strokeWidth="2"
          fill="none"
          initial={{
            pathLength: 0,
          }}
          style={{
            pathLength: pathLengths[2],
          }}
          transition={transition}
        />
        <motion.path
          d="M1040 0C1040 80 940 180 890 280C840 380 790 480 760 580C740 680 720 780 720 880C720 980 720 1080 720 1200"
          stroke="#BFDBFE"
          strokeWidth="2"
          fill="none"
          initial={{
            pathLength: 0,
          }}
          style={{
            pathLength: pathLengths[3],
          }}
          transition={transition}
        />
        <motion.path
          d="M940 0C940 60 890 160 840 260C790 360 760 460 740 560C720 660 720 760 720 860C720 960 720 1060 720 1200"
          stroke="#DBEAFE"
          strokeWidth="2"
          fill="none"
          initial={{
            pathLength: 0,
          }}
          style={{
            pathLength: pathLengths[4],
          }}
          transition={transition}
        />

        {/* Gaussian blur for the background paths */}

        <path
          d="M100 0C100 150 200 250 300 350C400 450 500 550 600 650C650 750 700 850 720 950C720 1050 720 1150 720 1200"
          stroke="#3B82F6"
          strokeWidth="2"
          fill="none"
          pathLength={1}
          opacity="0.3"
          filter="url(#blurMe)"
        />
        <path
          d="M200 0C200 120 300 220 400 320C500 420 550 520 600 620C650 720 680 820 720 920C720 1020 720 1120 720 1200"
          stroke="#60A5FA"
          strokeWidth="2"
          fill="none"
          pathLength={1}
          opacity="0.3"
          filter="url(#blurMe)"
        />
        <path
          d="M300 0C300 100 400 200 500 300C550 400 600 500 650 600C680 700 700 800 720 900C720 1000 720 1100 720 1200"
          stroke="#93C5FD"
          strokeWidth="2"
          fill="none"
          pathLength={1}
          opacity="0.3"
          filter="url(#blurMe)"
        />
        <path
          d="M400 0C400 80 500 180 550 280C600 380 650 480 680 580C700 680 720 780 720 880C720 980 720 1080 720 1200"
          stroke="#BFDBFE"
          strokeWidth="2"
          fill="none"
          pathLength={1}
          opacity="0.3"
          filter="url(#blurMe)"
        />
        <path
          d="M500 0C500 60 550 160 600 260C650 360 680 460 700 560C720 660 720 760 720 860C720 960 720 1060 720 1200"
          stroke="#DBEAFE"
          strokeWidth="2"
          fill="none"
          pathLength={1}
          opacity="0.3"
          filter="url(#blurMe)"
        />
        <path
          d="M1340 0C1340 150 1240 250 1140 350C1040 450 940 550 840 650C790 750 740 850 720 950C720 1050 720 1150 720 1200"
          stroke="#3B82F6"
          strokeWidth="2"
          fill="none"
          pathLength={1}
          opacity="0.3"
          filter="url(#blurMe)"
        />
        <path
          d="M1240 0C1240 120 1140 220 1040 320C940 420 890 520 840 620C790 720 760 820 720 920C720 1020 720 1120 720 1200"
          stroke="#60A5FA"
          strokeWidth="2"
          fill="none"
          pathLength={1}
          opacity="0.3"
          filter="url(#blurMe)"
        />
        <path
          d="M1140 0C1140 100 1040 200 940 300C890 400 840 500 790 600C760 700 740 800 720 900C720 1000 720 1100 720 1200"
          stroke="#93C5FD"
          strokeWidth="2"
          fill="none"
          pathLength={1}
          opacity="0.3"
          filter="url(#blurMe)"
        />
        <path
          d="M1040 0C1040 80 940 180 890 280C840 380 790 480 760 580C740 680 720 780 720 880C720 980 720 1080 720 1200"
          stroke="#BFDBFE"
          strokeWidth="2"
          fill="none"
          pathLength={1}
          opacity="0.3"
          filter="url(#blurMe)"
        />
        <path
          d="M940 0C940 60 890 160 840 260C790 360 760 460 740 560C720 660 720 760 720 860C720 960 720 1060 720 1200"
          stroke="#DBEAFE"
          strokeWidth="2"
          fill="none"
          pathLength={1}
          opacity="0.3"
          filter="url(#blurMe)"
        />

        <defs>
          <filter id="blurMe">
            <feGaussianBlur in="SourceGraphic" stdDeviation="5" />
          </filter>
        </defs>
      </svg>
    </div>
  );
};
