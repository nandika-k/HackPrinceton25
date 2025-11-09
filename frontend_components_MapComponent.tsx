/**
 * Geospatial Map Component using Leaflet
 * Shows device locations, alerts, and hazard zones
 */
import { useEffect, useRef } from 'react'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

interface TelemetryData {
  device_id: string
  location: { lat: number; lon: number }
  decision: string
  risk: number
}

interface AlertData extends TelemetryData {
  alert_type: string
}

interface MapComponentProps {
  telemetry: TelemetryData[]
  alerts: AlertData[]
}

export default function MapComponent({ telemetry, alerts }: MapComponentProps) {
  const mapRef = useRef<L.Map | null>(null)
  const mapContainerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!mapContainerRef.current || mapRef.current) return

    // Initialize map
    const map = L.map(mapContainerRef.current).setView([40.35, -74.65], 10) // Default: Princeton area

    // Add tile layer
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '© OpenStreetMap contributors',
      maxZoom: 19
    }).addTo(map)

    mapRef.current = map

    return () => {
      if (mapRef.current) {
        mapRef.current.remove()
        mapRef.current = null
      }
    }
  }, [])

  useEffect(() => {
    if (!mapRef.current) return

    const map = mapRef.current
    const markers: L.Marker[] = []

    // Add markers for all telemetry points
    telemetry.forEach((t) => {
      const color = t.decision === 'SOS' ? 'red' : t.decision === 'PREALERT' ? 'orange' : 'green'
      const icon = L.divIcon({
        className: 'custom-marker',
        html: `<div style="
          width: 12px;
          height: 12px;
          border-radius: 50%;
          background-color: ${color};
          border: 2px solid white;
          box-shadow: 0 0 4px rgba(0,0,0,0.5);
        "></div>`,
        iconSize: [12, 12],
        iconAnchor: [6, 6]
      })

      const marker = L.marker([t.location.lat, t.location.lon], { icon })
        .addTo(map)
        .bindPopup(`
          <strong>Device:</strong> ${t.device_id}<br/>
          <strong>Decision:</strong> ${t.decision}<br/>
          <strong>Risk:</strong> ${(t.risk * 100).toFixed(1)}%
        `)

      markers.push(marker)
    })

    // Add alert markers with larger icons
    alerts.forEach((alert) => {
      const icon = L.divIcon({
        className: 'alert-marker',
        html: `<div style="
          width: 20px;
          height: 20px;
          border-radius: 50%;
          background-color: ${alert.alert_type === 'SOS' ? 'red' : 'orange'};
          border: 3px solid white;
          box-shadow: 0 0 8px ${alert.alert_type === 'SOS' ? 'red' : 'orange'};
          animation: pulse 2s infinite;
        "></div>
        <style>
          @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
          }
        </style>`,
        iconSize: [20, 20],
        iconAnchor: [10, 10]
      })

      const marker = L.marker([alert.location.lat, alert.location.lon], { icon })
        .addTo(map)
        .bindPopup(`
          <strong>🚨 ${alert.alert_type} ALERT</strong><br/>
          <strong>Device:</strong> ${alert.device_id}<br/>
          <strong>Risk:</strong> ${(alert.risk * 100).toFixed(1)}%<br/>
          <strong>Location:</strong> ${alert.location.lat.toFixed(4)}, ${alert.location.lon.toFixed(4)}
        `)
        .openPopup()

      markers.push(marker)
    })

    // Fit map to show all markers
    if (markers.length > 0) {
      const group = new L.FeatureGroup(markers)
      map.fitBounds(group.getBounds().pad(0.1))
    }

    return () => {
      markers.forEach(marker => marker.remove())
    }
  }, [telemetry, alerts])

  return <div ref={mapContainerRef} style={{ width: '100%', height: '100%' }} />
}
