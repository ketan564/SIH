"use client"

import type React from "react"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import {
  ArrowLeft,
  TrendingUp,
  TrendingDown,
  DollarSign,
  BarChart3,
  Calendar,
  MapPin,
  AlertCircle,
  Target,
  Wheat,
  Apple,
  Carrot,
} from "lucide-react"
import { useRouter } from "next/navigation"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from "recharts"

interface CropPrice {
  name: string
  icon: React.ReactNode
  currentPrice: number
  previousPrice: number
  unit: string
  change: number
  changePercent: number
  trend: "up" | "down" | "stable"
  demand: "High" | "Medium" | "Low"
  supply: "High" | "Medium" | "Low"
}

interface MarketData {
  date: string
  price: number
  volume: number
}

export default function MarketTrendsPage() {
  const router = useRouter()
  const [selectedCrop, setSelectedCrop] = useState("tomato")
  const [timeRange, setTimeRange] = useState("7d")

  const cropPrices: CropPrice[] = [
    {
      name: "Tomato",
      icon: <Apple className="h-5 w-5" />,
      currentPrice: 45,
      previousPrice: 42,
      unit: "₹/kg",
      change: 3,
      changePercent: 7.1,
      trend: "up",
      demand: "High",
      supply: "Medium",
    },
    {
      name: "Wheat",
      icon: <Wheat className="h-5 w-5" />,
      currentPrice: 2150,
      previousPrice: 2200,
      unit: "₹/quintal",
      change: -50,
      changePercent: -2.3,
      trend: "down",
      demand: "Medium",
      supply: "High",
    },
    {
      name: "Onion",
      icon: <Carrot className="h-5 w-5" />,
      currentPrice: 35,
      previousPrice: 38,
      unit: "₹/kg",
      change: -3,
      changePercent: -7.9,
      trend: "down",
      demand: "Medium",
      supply: "Medium",
    },
    {
      name: "Rice",
      icon: <Wheat className="h-5 w-5" />,
      currentPrice: 3200,
      previousPrice: 3150,
      unit: "₹/quintal",
      change: 50,
      changePercent: 1.6,
      trend: "up",
      demand: "High",
      supply: "Medium",
    },
  ]

  const marketData: MarketData[] = [
    { date: "Mon", price: 42, volume: 1200 },
    { date: "Tue", price: 44, volume: 1350 },
    { date: "Wed", price: 43, volume: 1100 },
    { date: "Thu", price: 46, volume: 1450 },
    { date: "Fri", price: 45, volume: 1300 },
    { date: "Sat", price: 47, volume: 1600 },
    { date: "Sun", price: 45, volume: 1400 },
  ]

  const demandSupplyData = [
    { name: "Tomato", demand: 85, supply: 65 },
    { name: "Wheat", demand: 70, supply: 90 },
    { name: "Onion", demand: 60, supply: 70 },
    { name: "Rice", demand: 80, supply: 75 },
  ]

  const marketInsights = [
    {
      title: "Tomato Prices Rising",
      description: "Increased demand from processing industries driving prices up by 7%",
      type: "positive" as const,
      impact: "High",
    },
    {
      title: "Wheat Harvest Season",
      description: "New harvest arrivals causing temporary price dip, expected to stabilize",
      type: "neutral" as const,
      impact: "Medium",
    },
    {
      title: "Export Opportunity",
      description: "Strong international demand for rice varieties, good export window",
      type: "positive" as const,
      impact: "High",
    },
  ]

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case "up":
        return <TrendingUp className="h-4 w-4 text-green-600" />
      case "down":
        return <TrendingDown className="h-4 w-4 text-red-600" />
      default:
        return <BarChart3 className="h-4 w-4 text-gray-600" />
    }
  }

  const getTrendColor = (trend: string) => {
    switch (trend) {
      case "up":
        return "text-green-600"
      case "down":
        return "text-red-600"
      default:
        return "text-gray-600"
    }
  }

  const getDemandSupplyColor = (level: string) => {
    switch (level) {
      case "High":
        return "text-red-600 bg-red-100"
      case "Medium":
        return "text-yellow-600 bg-yellow-100"
      case "Low":
        return "text-green-600 bg-green-100"
      default:
        return "text-gray-600 bg-gray-100"
    }
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="bg-card border-b border-border px-4 py-4">
        <div className="flex items-center gap-3">
          <Button variant="ghost" size="sm" onClick={() => router.back()}>
            <ArrowLeft className="h-4 w-4" />
          </Button>
          <div>
            <h1 className="text-xl font-serif font-bold">Market Trends</h1>
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <MapPin className="h-4 w-4" />
              <span>Regional Markets</span>
            </div>
          </div>
        </div>
      </header>

      <main className="p-4 space-y-6">
        <Tabs defaultValue="prices" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="prices">Prices</TabsTrigger>
            <TabsTrigger value="analysis">Analysis</TabsTrigger>
            <TabsTrigger value="insights">Insights</TabsTrigger>
          </TabsList>

          <TabsContent value="prices" className="space-y-4">
            {/* Current Prices */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <DollarSign className="h-5 w-5 text-primary" />
                  Current Market Prices
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {cropPrices.map((crop, index) => (
                    <div key={index} className="flex items-center justify-between p-3 border border-border rounded-lg">
                      <div className="flex items-center gap-3">
                        <div className="p-2 bg-primary/10 rounded-lg">{crop.icon}</div>
                        <div>
                          <h3 className="font-semibold">{crop.name}</h3>
                          <p className="text-sm text-muted-foreground">
                            {crop.currentPrice} {crop.unit}
                          </p>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className={`flex items-center gap-1 ${getTrendColor(crop.trend)}`}>
                          {getTrendIcon(crop.trend)}
                          <span className="font-semibold">
                            {crop.change > 0 ? "+" : ""}
                            {crop.change} ({crop.changePercent > 0 ? "+" : ""}
                            {crop.changePercent}%)
                          </span>
                        </div>
                        <div className="flex gap-2 mt-1">
                          <Badge size="sm" className={getDemandSupplyColor(crop.demand)}>
                            D: {crop.demand}
                          </Badge>
                          <Badge size="sm" className={getDemandSupplyColor(crop.supply)}>
                            S: {crop.supply}
                          </Badge>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Price Chart */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  <span className="flex items-center gap-2">
                    <BarChart3 className="h-5 w-5 text-primary" />
                    Price Trends
                  </span>
                  <div className="flex gap-2">
                    <Select value={selectedCrop} onValueChange={setSelectedCrop}>
                      <SelectTrigger className="w-32">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="tomato">Tomato</SelectItem>
                        <SelectItem value="wheat">Wheat</SelectItem>
                        <SelectItem value="onion">Onion</SelectItem>
                        <SelectItem value="rice">Rice</SelectItem>
                      </SelectContent>
                    </Select>
                    <Select value={timeRange} onValueChange={setTimeRange}>
                      <SelectTrigger className="w-20">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="7d">7D</SelectItem>
                        <SelectItem value="1m">1M</SelectItem>
                        <SelectItem value="3m">3M</SelectItem>
                        <SelectItem value="1y">1Y</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={marketData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis />
                      <Tooltip />
                      <Line
                        type="monotone"
                        dataKey="price"
                        stroke="#ea580c"
                        strokeWidth={2}
                        dot={{ fill: "#ea580c" }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="analysis" className="space-y-4">
            {/* Demand vs Supply */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Target className="h-5 w-5 text-primary" />
                  Demand vs Supply Analysis
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={demandSupplyData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="demand" fill="#ea580c" name="Demand" />
                      <Bar dataKey="supply" fill="#f97316" name="Supply" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            {/* Market Indicators */}
            <Card>
              <CardHeader>
                <CardTitle>Market Indicators</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 gap-4">
                  <div className="text-center p-4 border border-border rounded-lg">
                    <div className="text-2xl font-bold text-green-600">↑ 12%</div>
                    <p className="text-sm text-muted-foreground">Overall Market Growth</p>
                  </div>
                  <div className="text-center p-4 border border-border rounded-lg">
                    <div className="text-2xl font-bold text-blue-600">₹2.4L</div>
                    <p className="text-sm text-muted-foreground">Avg. Daily Volume</p>
                  </div>
                  <div className="text-center p-4 border border-border rounded-lg">
                    <div className="text-2xl font-bold text-orange-600">85%</div>
                    <p className="text-sm text-muted-foreground">Market Efficiency</p>
                  </div>
                  <div className="text-center p-4 border border-border rounded-lg">
                    <div className="text-2xl font-bold text-purple-600">4.2</div>
                    <p className="text-sm text-muted-foreground">Price Volatility Index</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Best Selling Times */}
            <Card>
              <CardHeader>
                <CardTitle>Optimal Selling Windows</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex items-center justify-between p-3 bg-green-50 border border-green-200 rounded-lg">
                    <div>
                      <p className="font-semibold text-green-800">Tomatoes</p>
                      <p className="text-sm text-green-600">Peak season: Dec - Feb</p>
                    </div>
                    <Badge className="bg-green-100 text-green-800">Sell Now</Badge>
                  </div>
                  <div className="flex items-center justify-between p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
                    <div>
                      <p className="font-semibold text-yellow-800">Wheat</p>
                      <p className="text-sm text-yellow-600">Wait for post-harvest: Apr - May</p>
                    </div>
                    <Badge className="bg-yellow-100 text-yellow-800">Hold</Badge>
                  </div>
                  <div className="flex items-center justify-between p-3 bg-blue-50 border border-blue-200 rounded-lg">
                    <div>
                      <p className="font-semibold text-blue-800">Rice</p>
                      <p className="text-sm text-blue-600">Export window: Jan - Mar</p>
                    </div>
                    <Badge className="bg-blue-100 text-blue-800">Export Ready</Badge>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="insights" className="space-y-4">
            {/* Market Insights */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <AlertCircle className="h-5 w-5 text-primary" />
                  Market Insights
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {marketInsights.map((insight, index) => (
                    <div key={index} className="p-4 border border-border rounded-lg">
                      <div className="flex items-start justify-between mb-2">
                        <h3 className="font-semibold">{insight.title}</h3>
                        <Badge
                          className={
                            insight.type === "positive"
                              ? "bg-green-100 text-green-800"
                              : insight.type === "negative"
                                ? "bg-red-100 text-red-800"
                                : "bg-blue-100 text-blue-800"
                          }
                        >
                          {insight.impact} Impact
                        </Badge>
                      </div>
                      <p className="text-sm text-muted-foreground">{insight.description}</p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Trading Tips */}
            <Card>
              <CardHeader>
                <CardTitle>Trading Tips</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex items-start gap-3 p-3 bg-muted/50 rounded-lg">
                    <TrendingUp className="h-5 w-5 text-green-600 mt-0.5" />
                    <div>
                      <p className="font-medium">Monitor Weather Patterns</p>
                      <p className="text-sm text-muted-foreground">
                        Unexpected weather can cause 15-30% price fluctuations
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3 p-3 bg-muted/50 rounded-lg">
                    <Calendar className="h-5 w-5 text-blue-600 mt-0.5" />
                    <div>
                      <p className="font-medium">Track Seasonal Cycles</p>
                      <p className="text-sm text-muted-foreground">
                        Plan harvests around peak demand periods for better prices
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3 p-3 bg-muted/50 rounded-lg">
                    <BarChart3 className="h-5 w-5 text-orange-600 mt-0.5" />
                    <div>
                      <p className="font-medium">Diversify Crop Portfolio</p>
                      <p className="text-sm text-muted-foreground">
                        Reduce risk by growing multiple crops with different market cycles
                      </p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Price Alerts */}
            <Card>
              <CardHeader>
                <CardTitle>Set Price Alerts</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <Button variant="outline" className="w-full justify-start bg-transparent">
                    <AlertCircle className="h-4 w-4 mr-2" />
                    Alert when Tomato reaches ₹50/kg
                  </Button>
                  <Button variant="outline" className="w-full justify-start bg-transparent">
                    <AlertCircle className="h-4 w-4 mr-2" />
                    Alert when Wheat drops below ₹2000/quintal
                  </Button>
                  <Button className="w-full">+ Add New Alert</Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </main>

      {/* Spacer for bottom navigation */}
      <div className="h-16"></div>
    </div>
  )
}
