export default function Drop() {
  return(
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
          </main>
      </div>
  )
}
