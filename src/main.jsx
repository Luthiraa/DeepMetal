import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import Work from './Work.jsx'
import Drop from './drop.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <Work />
    <Drop/>
    <App/>
  </StrictMode>,
)
