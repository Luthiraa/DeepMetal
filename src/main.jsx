import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import Work from './Work.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <Work />
  </StrictMode>,
)
