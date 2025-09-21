"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import Link from "next/link"
import WeatherCard from '../components/WeatherCard';

import {
  Cloud,
  Thermometer,
  Droplets,
  Wind,
  Camera,
  MessageCircle,
  TrendingUp,
  MapPin,
  Sun,
  CloudRain,
  Leaf,
  Star,
  ArrowUp,
  ArrowDown,
  AlertTriangle,
} from "lucide-react"

export default function HomePage() {
  const [headerLocation, setHeaderLocation] = useState<string>("Detecting location...");
  const [location, setLocation] = useState<string>("Getting location...")
  const [weather, setWeather] = useState({
    temperature: 28,
    humidity: 65,
    windSpeed: 12,
    condition: "Partly Cloudy",
  })

  const [bestCrops, setBestCrops] = useState([
    {
      name: "Tomatoes",
      suitability: 95,
      reason: "Perfect weather conditions",
      marketPrice: "₹45/kg",
      trend: "up",
      priceChange: "+12%",
    },
    {
      name: "Wheat",
      suitability: 88,
      reason: "Good soil moisture",
      marketPrice: "₹2,100/quintal",
      trend: "up",
      priceChange: "+5%",
    },
    {
      name: "Cotton",
      suitability: 82,
      reason: "Favorable temperature",
      marketPrice: "₹6,800/quintal",
      trend: "down",
      priceChange: "-3%",
    },
  ])

  useEffect(() => {
    // Simulate getting user location
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          setLocation("Current Location")
        },
        () => {
          setLocation("Location unavailable")
        },
      )
    }
  }, [])

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="bg-card border-b border-border px-4 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-serif font-bold text-primary">Agri-Sage</h1>
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <MapPin className="h-4 w-4" />
              <span>{headerLocation}</span>
            </div>
          </div>
          <Link href="/chat">
            <Button variant="outline" size="sm">
              <MessageCircle className="h-4 w-4 mr-2" />
              AI Assistant
            </Button>
          </Link>
        </div>
      </header>

      {/* Main Content */}
      <main className="p-4 space-y-6">
        <Card className="border-primary/20">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Leaf className="h-5 w-5 text-primary" />
              Best Crops for You
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {bestCrops.map((crop, index) => (
              <div key={crop.name} className="flex items-center justify-between p-3 bg-muted/30 rounded-lg">
                <div className="flex items-center gap-3">
                  <div className="flex items-center gap-2">
                    <span className="text-lg font-bold text-primary">#{index + 1}</span>
                    <Star className="h-4 w-4 text-yellow-500 fill-current" />
                  </div>
                  <div>
                    <p className="font-semibold">{crop.name}</p>
                    <p className="text-sm text-muted-foreground">{crop.reason}</p>
                    <div className="flex items-center gap-2 mt-1">
                      <Badge variant="outline" className="text-xs border-primary text-primary">
                        {crop.suitability}% match
                      </Badge>
                      <span className="text-sm font-medium">{crop.marketPrice}</span>
                      <div
                        className={`flex items-center gap-1 text-xs ${
                          crop.trend === "up" ? "text-green-600" : "text-red-600"
                        }`}
                      >
                        {crop.trend === "up" ? <ArrowUp className="h-3 w-3" /> : <ArrowDown className="h-3 w-3" />}
                        <span>{crop.priceChange}</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ))}
            <Button className="w-full bg-transparent" variant="outline">
              View Detailed Analysis
            </Button>
          </CardContent>
        </Card> 

        <WeatherCard onLocation={setHeaderLocation}/>


        {/* Quick Actions */}
        <div className="grid grid-cols-2 gap-4">
          <Link href="/disease-detection">
            <Card className="cursor-pointer hover:shadow-md transition-shadow">
              <CardContent className="p-6 text-center">
                <div className="p-3 bg-primary/10 rounded-full w-fit mx-auto mb-3">
                  <Camera className="h-6 w-6 text-primary" />
                </div>
                <h3 className="font-semibold mb-1">Disease Detection</h3>
                <p className="text-sm text-muted-foreground">Scan crop diseases instantly</p>
              </CardContent>
            </Card>
          </Link>

          <Link href="/market-trends">
            <Card className="cursor-pointer hover:shadow-md transition-shadow">
              <CardContent className="p-6 text-center">
                <div className="p-3 bg-primary/10 rounded-full w-fit mx-auto mb-3">
                  <TrendingUp className="h-6 w-6 text-primary" />
                </div>
                <h3 className="font-semibold mb-1">Market Trends</h3>
                <p className="text-sm text-muted-foreground">Check crop prices</p>
              </CardContent>
            </Card>
          </Link>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Today's Recommendations</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-start gap-3 p-3 bg-green-50 border border-green-200 rounded-lg">
              <Sun className="h-5 w-5 text-green-600 mt-0.5" />
              <div>
                <p className="font-medium text-green-800">Perfect day for planting tomatoes</p>
                <p className="text-sm text-green-600">Soil temperature and moisture levels are optimal</p>
              </div>
            </div>
            <div className="flex items-start gap-3 p-3 bg-blue-50 border border-blue-200 rounded-lg">
              <CloudRain className="h-5 w-5 text-blue-600 mt-0.5" />
              <div>
                <p className="font-medium text-blue-800">Rain expected tomorrow</p>
                <p className="text-sm text-blue-600">Consider postponing irrigation and fertilizer application</p>
              </div>
            </div>
            <div className="flex items-start gap-3 p-3 bg-orange-50 border border-orange-200 rounded-lg">
              <AlertTriangle className="h-5 w-5 text-orange-600 mt-0.5" />
              <div>
                <p className="font-medium text-orange-800">Pest alert: Aphids detected nearby</p>
                <p className="text-sm text-orange-600">Monitor your crops and consider preventive measures</p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Recent Activity */}
        <Card>
          <CardHeader>
            <CardTitle>Recent Activity</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="flex items-center justify-between py-2">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-green-100 rounded-lg">
                    <Camera className="h-4 w-4 text-green-600" />
                  </div>
                  <div>
                    <p className="font-medium">Tomato leaf scanned</p>
                    <p className="text-sm text-muted-foreground">2 hours ago</p>
                  </div>
                </div>
                <Badge variant="outline" className="text-green-600">
                  Healthy
                </Badge>
              </div>
              <div className="flex items-center justify-between py-2">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-blue-100 rounded-lg">
                    <MessageCircle className="h-4 w-4 text-blue-600" />
                  </div>
                  <div>
                    <p className="font-medium">Asked about fertilizers</p>
                    <p className="text-sm text-muted-foreground">Yesterday</p>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </main>

      {/* Bottom Navigation */}
      <nav className="fixed bottom-0 left-0 right-0 bg-card border-t border-border">
        <div className="grid grid-cols-4 gap-1">
          <Link href="/weather" className="contents">
            <Button variant="ghost" className="h-16 flex-col gap-1 rounded-none">
              <Cloud className="h-5 w-5" />
              <span className="text-xs">Weather</span>
            </Button>
          </Link>
          <Link href="/disease-detection" className="contents">
            <Button variant="ghost" className="h-16 flex-col gap-1 rounded-none">
              <Camera className="h-5 w-5" />
              <span className="text-xs">Scan</span>
            </Button>
          </Link>
          <Link href="/chat" className="contents">
            <Button variant="ghost" className="h-16 flex-col gap-1 rounded-none">
              <MessageCircle className="h-5 w-5" />
              <span className="text-xs">Chat</span>
            </Button>
          </Link>
          <Link href="/market-trends" className="contents">
            <Button variant="ghost" className="h-16 flex-col gap-1 rounded-none">
              <TrendingUp className="h-5 w-5" />
              <span className="text-xs">Market</span>
            </Button>
          </Link>
        </div>
      </nav>

      {/* Spacer for bottom navigation */}
      <div className="h-16"></div>
    </div>
  )
}
